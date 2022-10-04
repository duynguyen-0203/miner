import csv
import random
from typing import List, Tuple

from transformers import PreTrainedTokenizer

from src import constants
from src.entities import Dataset, News


class Reader:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_title_length: int, max_sapo_length: int, user2id: dict,
                 category2id: dict, max_his_click: int, npratio: int = None):
        self._tokenizer = tokenizer
        self._max_title_length = max_title_length
        self._max_sapo_length = max_sapo_length
        self._user2id = user2id
        self._category2id = category2id
        self._max_his_click = max_his_click
        self._npratio = npratio

    def read_train_dataset(self, data_name: str, news_path: str, behaviors_path: str) -> Dataset:
        dataset, news_dataset = self._read(data_name, news_path)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                self._parse_train_line(i, line, news_dataset, dataset)

        return dataset

    def read_eval_dataset(self, data_name: str, news_path: str, behaviors_path: str) -> Dataset:
        dataset, news_dataset = self._read(data_name, news_path)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                self._parse_eval_line(i, line, news_dataset, dataset)

        return dataset

    def _read(self, data_name: str, news_path: str) -> Tuple[Dataset, dict]:
        dataset = Dataset(data_name, self._tokenizer, self._category2id)
        news_dataset = self._read_news_info(news_path, dataset)

        return dataset, news_dataset

    def _read_news_info(self, news_path: str, dataset: Dataset) -> dict:
        r"""
        Read news information

        Args:
            news_path: path to TSV file containing all news information.
            dataset: Dataset object.

        Returns:
            A dictionary
        """
        pad_news_obj = dataset.create_news(
            [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id],
            [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id], self._category2id['pad'])
        news_dataset = {'pad': pad_news_obj}
        with open(news_path, mode='r', encoding='utf-8', newline='') as f:
            news_tsv = csv.reader(f, delimiter='\t')
            for line in news_tsv:
                title_encoding = self._tokenizer.encode(line[constants.TITLE], add_special_tokens=True, truncation=True,
                                                        max_length=self._max_title_length)
                category_id = self._category2id.get(line[constants.CATEGORY], self._category2id['unk'])
                sapo_encoding = self._tokenizer.encode(line[constants.SAPO], add_special_tokens=True, truncation=True,
                                                       max_length=self._max_sapo_length)
                news = dataset.create_news(title_encoding, sapo_encoding, category_id)
                news_dataset[line[constants.NEWS_ID]] = news

        return news_dataset

    def _parse_train_line(self, impression_id, line, news_dataset, dataset):
        r"""
        Parse a line of the training dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])
        history_clicked = [news_dataset[news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        pos_news = [news_dataset[news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1']
        neg_news = [news_dataset[news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '0']
        for news in pos_news:
            label = [1] + [0] * self._npratio
            list_news = [news] + sample_news(neg_news, self._npratio, news_dataset['pad'])
            impression_news = list(zip(list_news, label))
            random.shuffle(impression_news)
            list_news, label = zip(*impression_news)
            impression = dataset.create_impression(impression_id, user_id, list_news, label)
            dataset.add_sample(user_id, history_clicked, impression)

    def _parse_eval_line(self, impression_id, line, news_dataset, dataset):
        r"""
        Parse a line of the evaluation dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])
        history_clicked = [news_dataset[news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        for behavior in line[constants.BEHAVIOR].split():
            news_id, label = behavior.split('-')
            impression = dataset.create_impression(impression_id, user_id, [news_dataset[news_id]], [int(label)])
            dataset.add_sample(user_id, history_clicked, impression)


def sample_news(list_news: List[News], num_news: int, pad: News) -> List:
    if len(list_news) >= num_news:
        return random.sample(list_news, k=num_news)
    else:
        return list_news + [pad] * (num_news - len(list_news))
