from collections import OrderedDict
from typing import List

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from src import utils


class News:
    def __init__(self, news_id: str, title: List[int], sapo: List[int], category: int):
        self._news_id = news_id
        self._title = title
        self._sapo = sapo
        self._category = category

    @property
    def news_id(self) -> int:
        return self._news_id

    @property
    def title(self) -> List[int]:
        return self._title

    @property
    def sapo(self) -> List[int]:
        return self._sapo

    @property
    def category(self) -> int:
        return self._category


class Impression:
    def __init__(self, impression_id: int, user_id: int, news: List[News], label: List[int]):
        self._impression_id = impression_id
        self._user_id = user_id
        self._news = news
        self._label = label

    @property
    def impression_id(self):
        return self._impression_id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def news(self) -> List[News]:
        return self._news

    @property
    def label(self):
        return self._label


class Sample:
    def __init__(self, sample_id: int, user_id: int, clicked_news: List[News], impression: Impression):
        self._sample_id = sample_id
        self._user_id = user_id
        self._clicked_news = clicked_news
        self._impression = impression

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def clicked_news(self) -> List[News]:
        return self._clicked_news

    @property
    def impression(self) -> Impression:
        return self._impression


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict):
        super().__init__()
        self._name = data_name
        self._samples = OrderedDict()
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._category2id = category2id

        self._news_id = 0
        self._id = 0

    def set_mode(self, mode: str):
        self._mode = mode

    def create_news(self, title: List[int], sapo: List[int], category: int) -> News:
        news = News(self._news_id, title, sapo, category)
        self._news_id += 1

        return news

    @staticmethod
    def create_impression(impression_id: int, user_id: int, news: List[News], label: List[int]) -> Impression:
        impression = Impression(impression_id, user_id, news, label)

        return impression

    def add_sample(self, user_id: int, clicked_news: List[News], impression: Impression):
        sample = Sample(self._id, user_id, clicked_news, impression)
        self._samples[self._id] = sample
        self._id += 1

    @property
    def samples(self) -> List[Sample]:
        return list(self._samples.values())

    @property
    def news_count(self) -> int:
        return self._news_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        sample = self.samples[i]

        if self._mode == Dataset.TRAIN_MODE:
            return create_train_sample(sample, self._tokenizer, self._category2id)
        else:
            return create_eval_sample(sample, self._tokenizer, self._category2id)


def _create_sample(sample: Sample, tokenizer: PreTrainedTokenizer, category2id: dict) -> dict:
    title_clicked_news_encoding = [news.title for news in sample.clicked_news]
    sapo_clicked_news_encoding = [news.sapo for news in sample.clicked_news]
    category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    title_impression_encoding = [news.title for news in sample.impression.news]
    sapo_impression_encoding = [news.sapo for news in sample.impression.news]
    category_impression_encoding = [news.category for news in sample.impression.news]

    # Create tensor
    impression_id = torch.tensor(sample.impression.impression_id)
    title_clicked_news_encoding = utils.padded_stack(title_clicked_news_encoding, padding=tokenizer.pad_token_id)
    sapo_clicked_news_encoding = utils.padded_stack(sapo_clicked_news_encoding, padding=tokenizer.pad_token_id)
    category_clicked_news_encoding = torch.tensor(category_clicked_news_encoding)
    his_mask = (category_clicked_news_encoding != category2id['pad'])
    his_title_mask = (title_clicked_news_encoding != tokenizer.pad_token_id)
    his_sapo_mask = (sapo_clicked_news_encoding != tokenizer.pad_token_id)

    title_impression_encoding = utils.padded_stack(title_impression_encoding, padding=tokenizer.pad_token_id)
    sapo_impression_encoding = utils.padded_stack(sapo_impression_encoding, padding=tokenizer.pad_token_id)
    category_impression_encoding = torch.tensor(category_impression_encoding)
    title_mask = (title_impression_encoding != tokenizer.pad_token_id)
    sapo_mask = (sapo_impression_encoding != tokenizer.pad_token_id)

    label = torch.tensor(sample.impression.label)

    return dict(his_title=title_clicked_news_encoding, his_title_mask=his_title_mask,
                his_sapo=sapo_clicked_news_encoding, his_sapo_mask=his_sapo_mask,
                his_category=category_clicked_news_encoding, his_mask=his_mask, title=title_impression_encoding,
                title_mask=title_mask, sapo=sapo_impression_encoding, sapo_mask=sapo_mask,
                category=category_impression_encoding, impression_id=impression_id, label=label)


def create_train_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int) -> dict:
    return _create_sample(sample, tokenizer, num_category)


def create_eval_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int) -> dict:
    return _create_sample(sample, tokenizer, num_category)
