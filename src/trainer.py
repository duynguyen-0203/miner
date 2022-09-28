import json
import math
import time
from typing import List

import torch
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, RobertaConfig

from src import utils
from src.base_trainer import BaseTrainer
from src.entities import Dataset
from src.evaluation import FastEvaluator, SlowEvaluator
from src.model.model import Miner
from src.model.news_encoder import NewsEncoder
from src.loss import Loss
from src.reader import Reader


class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)

        with open(args.user2id_path, mode='r', encoding='utf-8') as f:
            self._user2id = json.load(f)
        with open(args.category2id_path, mode='r', encoding='utf-8') as f:
            self._category2id = json.load(f)

        if 'fp16' in args:
            if args.fp16:
                self.scaler = GradScaler()
            else:
                self.scaler = None

    def train(self):
        args = self.args
        self._log_arguments()
        self._logger.info(f'Model: {args.model_name}')

        # Read pretrained embedding (if any)
        if args.category_embed_path is not None:
            category_embed = utils.load_embed(args.category_embed_path)
        else:
            category_embed = None

        # Read data
        reader = Reader(tokenizer=self._tokenizer, max_title_length=args.max_title_length,
                        max_sapo_length=args.max_sapo_length, user2id=self._user2id, category2id=self._category2id,
                        max_his_click=args.his_length, npratio=args.npratio)
        train_dataset = reader.read_train_dataset(args.data_name, args.train_news_path, args.train_behaviors_path)
        train_dataset.set_mode(Dataset.TRAIN_MODE)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False,
                                      num_workers=args.dataloader_num_workers, collate_fn=self._collate_fn,
                                      drop_last=args.dataloader_drop_last, pin_memory=args.dataloader_pin_memory)
        if args.fast_eval:
            eval_dataset = reader.read_train_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        else:
            eval_dataset = reader.read_eval_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        self._log_dataset(train_dataset, eval_dataset)

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if args.max_steps is not None:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps %
                                                                                  num_update_steps_per_epoch > 0)
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)

        self._logger.info('------------- Start training -------------')
        self._logger.info(f'Num epochs: {num_train_epochs}')
        self._logger.info(f'Updates per epoch: {num_update_steps_per_epoch}')
        self._logger.info(f'Updates total: {max_steps}')
        self._logger.info(f'Total train batch size: {total_train_batch_size}')
        self._logger.info(f'Gradient accumulation steps: {args.gradient_accumulation_steps}')

        # Create model
        config = RobertaConfig.from_pretrained(args.pretrained_embedding)
        news_encoder = NewsEncoder.from_pretrained(args.pretrained_embedding, config=config,
                                                   apply_reduce_dim=args.apply_reduce_dim, use_sapo=args.use_sapo,
                                                   dropout=args.dropout, freeze_transformer=args.freeze_transformer,
                                                   word_embed_dim=args.word_embed_dim, combine_type=args.combine_type,
                                                   lstm_num_layers=args.lstm_num_layers, lstm_dropout=args.lstm_dropout)
        model = Miner(news_encoder=news_encoder, use_category_bias=args.use_category_bias,
                      num_context_codes=args.num_context_codes, context_code_dim=args.context_code_dim,
                      score_type=args.score_type, dropout=args.dropout, num_category=len(self._category2id),
                      category_embed_dim=args.category_embed_dim, category_pad_token_id=self._category2id['pad'],
                      category_embed=category_embed)
        model.to(self._device)
        model.zero_grad(set_to_none=True)

        # Create optimizer and scheduler
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=self._get_warmup_steps(max_steps),
                                                                 num_training_steps=max_steps)
        loss_calculator = self._create_loss()

        best_valid_loss = float('inf')
        best_auc_score = 0.0
        global_step = 0
        global_iteration = 0
        logging_loss = 0.0

        for epoch in range(args.num_train_epochs):
            epoch_start_time = time.time()
            eval_time = 0.0
            self._logger.info(f'--------------- EPOCH {epoch} ---------------')
            steps_in_epoch = len(train_dataloader)
            epoch_loss = 0.0
            accumulation_factor = (self.args.gradient_accumulation_steps
                                   if steps_in_epoch > self.args.gradient_accumulation_steps else steps_in_epoch)

            for step, batch in tqdm(enumerate(train_dataloader), total=steps_in_epoch, desc=f'Train epoch {epoch}'):
                batch_loss = self._train_step(batch, model, loss_calculator, accumulation_factor)
                logging_loss += batch_loss
                epoch_loss += batch_loss
                self._log_train_step(scheduler, batch_loss, global_iteration)
                if not (not ((global_iteration + 1) % args.gradient_accumulation_steps == 0) and not (
                        args.gradient_accumulation_steps >= steps_in_epoch == (step + 1))):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if args.fp16:
                            self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Optimizer step
                    optimizer_was_run = True
                    if args.fp16:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        optimizer.step()

                    if optimizer_was_run:
                        scheduler.step()

                    model.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.logging_steps == 0:
                        logging_loss /= (args.logging_steps * accumulation_factor)
                        self._log_train(scheduler, logging_loss, global_step, epoch)
                        logging_loss = 0.0

                    if global_step % args.eval_steps == 0:
                        eval_start_time = time.time()
                        valid_loss, scores = self._eval(model, eval_dataset, loss_calculator, metrics=args.metrics)
                        self._log_eval(global_step, valid_loss, scores)

                        if 'loss' in self.args.evaluation_info and valid_loss < best_valid_loss:
                            self._logger.info(f'Best loss updates from {best_valid_loss} to {valid_loss}, '
                                              f'at global step {global_step}')
                            best_valid_loss = valid_loss
                            self._save_model(model, optimizer, scheduler, flag='bestLossModel')
                        if 'metrics' in self.args.evaluation_info and scores['auc'] > best_auc_score:
                            self._logger.info(f'Best AUC score updates from {best_auc_score} to {scores["auc"]}, '
                                              f'at global step {global_step}')
                            best_auc_score = scores['auc']
                            self._save_model(model, optimizer, scheduler, flag='bestAucModel')
                        eval_time += time.time() - eval_start_time
                global_iteration += 1

            # Evaluation at the end of each epoch
            eval_start_time = time.time()
            valid_loss, scores = self._eval(model, eval_dataset, loss_calculator, metrics=args.metrics)
            train_loss = epoch_loss / steps_in_epoch
            self._log_epoch(train_loss, valid_loss, scores, epoch)
            if 'loss' in self.args.evaluation_info and valid_loss < best_valid_loss:
                self._logger.info(f'Best loss updates from {best_valid_loss} to {valid_loss}, at epoch {epoch}')
                best_valid_loss = valid_loss
                self._save_model(model, optimizer, scheduler, flag='bestLossModel')
            if 'metrics' in self.args.evaluation_info and scores['auc'] > best_auc_score:
                self._logger.info(f'Best AUC score updates from {best_auc_score} to {scores["auc"]}, at epoch {epoch}')
                best_auc_score = scores['auc']
                self._save_model(model, optimizer, scheduler, flag='bestAucModel')
            eval_time += time.time() - eval_start_time

            # Log running time
            epoch_end_time = time.time()
            self._logger.info(f'Total running time of epoch: {round(epoch_end_time - epoch_start_time, ndigits=4)} (s)')
            self._logger.info(f'Total training time of epoch: '
                              f'{round(epoch_end_time - epoch_start_time - eval_time, ndigits=4)} (s)')

        # Save final model
        self._save_model(model, optimizer, scheduler, flag='finalModel')
        self._logger.info('---  Finish training!!!  ---')

    def eval(self):
        args = self.args

        # Load model
        model = self._load_model(args.saved_model_path)
        model.to(self._device)

        # Read eval dataset
        reader = Reader(tokenizer=self._tokenizer, max_title_length=args.max_title_length,
                        max_sapo_length=args.max_sapo_length, user2id=self._user2id, category2id=self._category2id,
                        max_his_click=args.max_his_click)
        dataset = reader.read_eval_dataset(args.data_name, args.news_path, args.eval_behaviors_path)
        self._logger.info(f'Model: {self.args.model_name}')
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Test dataset: {len(dataset)} samples')

        # Evaluation
        loss_calculator = self._create_loss()
        self._logger.info('----------------  Evaluation phrase  ----------------')
        loss, scores = self._eval(model, dataset, loss_calculator, metrics=args.metrics,
                                  save_result=args.save_eval_result)
        self._logger.info('Loss {}'.format(loss))
        for metric in args.metrics:
            self._logger.info(f'Metric {metric}: {scores[metric]}')

    def _train_step(self, batch, model, loss_calculator, accumulation_factor: int):
        model.train()
        batch = utils.to_device(batch, self._device)
        if self.args.fp16:
            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                poly_attn, logits = self._forward_step(model, batch)
                loss = loss_calculator.compute(poly_attn, logits, batch['label'])
                loss = loss / accumulation_factor
            self.scaler.scale(loss).backward()
        else:
            poly_attn, logits = self._forward_step(model, batch)
            loss = loss_calculator.compute(poly_attn, logits, batch['label'])
            loss = loss / accumulation_factor
            loss.backward()

        return (loss * accumulation_factor).item()

    def _eval(self, model, dataset, loss_calculator, metrics: List[str], save_result: bool = False):
        model.eval()
        dataset.set_mode(Dataset.EVAL_MODE)
        if self.args.fast_eval:
            evaluator = FastEvaluator(dataset)
        else:
            evaluator = SlowEvaluator(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                num_workers=self.args.dataloader_num_workers, collate_fn=self._collate_fn,
                                drop_last=False)
        total_loss = 0.0
        total_pos_example = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation phase'):
                batch = utils.to_device(batch, self._device)
                poly_attn, logits = self._forward_step(model, batch)
                if 'loss' in self.args.evaluation_info:
                    batch_loss = loss_calculator.compute_eval_loss(poly_attn, logits, batch['label'])
                    total_loss += batch_loss
                    total_pos_example += batch['label'].sum().item()
                if 'metrics' in self.args.evaluation_info:
                    evaluator.eval_batch(logits, batch['impression_id'])

        if 'loss' in self.args.evaluation_info:
            loss = total_loss / total_pos_example
        else:
            loss = None
        if 'metrics' in self.args.evaluation_info:
            scores = evaluator.compute_scores(metrics, save_result, self._path)
        else:
            scores = None

        return loss, scores

    @staticmethod
    def _create_loss():
        criterion = nn.CrossEntropyLoss(reduction='mean')
        loss_calculator = Loss(criterion)

        return loss_calculator

    def _collate_fn(self, batch):
        padded_batch = dict()
        keys = batch[0].keys()
        for key in keys:
            samples = [s[key] for s in batch]
            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                if key in ['his_title', 'title', 'his_sapo', 'sapo']:
                    padded_batch[key] = utils.padded_stack(samples, padding=self._tokenizer.pad_token_id)
                elif key in ['his_category', 'category']:
                    padded_batch[key] = utils.padded_stack(samples, padding=self._category2id['pad'])
                else:
                    padded_batch[key] = utils.padded_stack(samples, padding=0)

        return padded_batch

    def _get_optimizer_params(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                             'weight_decay': self.args.weight_decay},
                            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0}]

        return optimizer_params

    @staticmethod
    def _forward_step(model, batch):
        poly_attn, logits = model(title=batch['title'], title_mask=batch['title_mask'], his_title=batch['his_title'],
                                  his_title_mask=batch['his_title_mask'], his_mask=batch['his_mask'],
                                  sapo=batch['sapo'], sapo_mask=batch['sapo_mask'], his_sapo=batch['his_sapo'],
                                  his_sapo_mask=batch['his_sapo_mask'], category=batch['category'],
                                  his_category=batch['his_category'])
        return poly_attn, logits
