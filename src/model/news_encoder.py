from abc import ABC
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel


class NewsEncoder(ABC, RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, apply_reduce_dim: bool, use_sapo: bool, dropout: float,
                 freeze_transformer: bool, word_embed_dim: Union[int, None] = None,
                 combine_type: Union[str, None] = None, lstm_num_layers: Union[int, None] = None,
                 lstm_dropout: Union[float, None] = None):
        r"""
        Initialization

        Args:
            config: the configuration of a ``RobertaModel``.
            apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
            use_sapo: whether to use sapo embedding or not.
            dropout: dropout value.
            freeze_transformer: whether to freeze Roberta weight or not.
            word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
            combine_type: method to combine news information.
            lstm_num_layers: number of recurrent layers in LSTM.
            lstm_dropout: dropout value in LSTM.
        """
        super().__init__(config)
        self.roberta = RobertaModel(config)
        if freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.apply_reduce_dim = apply_reduce_dim

        if self.apply_reduce_dim:
            assert word_embed_dim is not None
            self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
            self.word_embed_dropout = nn.Dropout(dropout)
            self._embed_dim = word_embed_dim
        else:
            self._embed_dim = config.hidden_size

        self.use_sapo = use_sapo
        if self.use_sapo:
            assert combine_type is not None
            self.combine_type = combine_type
            if self.combine_type == 'linear':
                self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
            elif self.combine_type == 'lstm':
                self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
                                    num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout,
                                    bidirectional=True)
                self._embed_dim = (self._embed_dim // 2) * 2

        self.init_weights()

    def forward(self, title_encoding: Tensor, title_attn_mask: Tensor, sapo_encoding: Union[Tensor, None] = None,
                sapo_attn_mask: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        news_info = []
        # Title encoder
        title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        title_repr = title_word_embed[:, 0, :]
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            title_repr = self.word_embed_dropout(title_repr)
        news_info.append(title_repr)

        # Sapo encoder
        if self.use_sapo:
            sapo_word_embed = self.roberta(input_ids=sapo_encoding, attention_mask=sapo_attn_mask)[0]
            sapo_repr = sapo_word_embed[:, 0, :]
            if self.apply_reduce_dim:
                sapo_repr = self.reduce_dim(sapo_repr)
                sapo_repr = self.word_embed_dropout(sapo_repr)
            news_info.append(sapo_repr)

            if self.combine_type == 'linear':
                news_info = torch.cat(news_info, dim=1)

                return self.linear_combine(news_info)
            elif self.combine_type == 'lstm':
                news_info = torch.cat(news_info, dim=1)
                news_repr, _ = self.lstm(news_info)

                return news_repr
        else:
            return title_repr

    @property
    def embed_dim(self):
        return self._embed_dim
