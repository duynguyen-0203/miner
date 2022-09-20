from abc import ABC
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel

from src.utils import pairwise_cosine_similarity


class Miner(ABC, RobertaPreTrainedModel):
    r"""
    Implementation of Multi-interest matching network for news recommendation. Please see the paper in
    https://aclanthology.org/2022.findings-acl.29.pdf.
    """
    def __init__(self, config: RobertaConfig, apply_reduce_dim: bool, reduce_dim: int, use_category_bias: bool,
                 num_category: int, category_embed_dim: int, category_pad_token_id: int, num_context_codes: int,
                 context_code_dim: int, score_type: str, dropout: float):
        r"""
        Initialization

        Args:
            config: The configuration of a Roberta Model
            apply_reduce_dim: Whether to reduce the dimension of word embeddings
            reduce_dim: The size of each word embedding vector if ``apply_reduce_dim``
            use_category_bias: Whether to use Category-aware attention weighting
            num_category: The size of the dictionary of categories
            category_embed_dim: The size of each category embedding vector
            category_pad_token_id: ID of the padding token type in the category vocabulary
            num_context_codes: The number of attention vectors ``K``
            context_code_dim: The number of features in a context code
            score_type: The ways to aggregate the ``K`` matching scores as a final user click score ('max', 'mean' or
                'weighted')
            dropout: Dropout value
        """
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.apply_reduce_dim = apply_reduce_dim
        if self.apply_reduce_dim:
            self.reduce_embed_dim = nn.Linear(in_features=config.hidden_size, out_features=reduce_dim)
            self._news_embed_dim = reduce_dim
        else:
            self._news_embed_dim = config.hidden_size

        self.use_category_bias = use_category_bias
        if self.use_category_bias:
            self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=category_embed_dim,
                                                   padding_idx=category_pad_token_id)
            self.category_dropout = nn.Dropout(dropout)
            self.category_weight = nn.Parameter(torch.randn(1))

        self.poly_attn = PolyAttention(in_embed_dim=self._news_embed_dim, num_context_codes=num_context_codes,
                                       context_code_dim=context_code_dim)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self._news_embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, history_encoding: Tensor, history_attn_mask: Tensor, history_category_encoding: Tensor,
                history_mask: Tensor, candidate_encoding: Tensor, candidate_attn_mask: Tensor,
                candidate_category_encoding: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagation

        Args:
            history_encoding: tensor of shape ``(batch_size, his_length, seq_length)``
            history_attn_mask: tensor of shape ``(batch_size, his_length, seq_length)``
            history_category_encoding: tensor of shape ``(batch_size, his_length)``
            history_mask: tensor of shape ``(batch_size, his_length)``
            candidate_encoding: tensor of shape ``(batch_size, num_candidates, seq_length)``
            candidate_attn_mask: tensor of shape ``(batch_size, num_candidates, seq_length)``
            candidate_category_encoding: tensor of shape ``(batch_size, num_candidates)``

        Returns:
            tuple:
                - multi_user_interest: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
                - matching_scores: tensor of shape ``(batch_size, num_candidates)``
        """
        batch_size = history_encoding.shape[0]
        his_length = history_encoding.shape[1]
        num_candidates = candidate_encoding.shape[1]

        # Representation of candidate news
        candidate_encoding = candidate_encoding.view(batch_size * num_candidates, -1)
        candidate_attn_mask = candidate_attn_mask.view(batch_size * num_candidates, -1)
        candidate_embedding = self.roberta(input_ids=candidate_encoding, attention_mask=candidate_attn_mask)[0]

        candidate_repr = candidate_embedding[:, 0, :]
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)

        # Representation of history clicked news
        history_encoding = history_encoding.view(batch_size * his_length, -1)
        history_attn_mask = history_attn_mask.view(batch_size * his_length, -1)
        history_embedding = self.roberta(input_ids=history_encoding, attention_mask=history_attn_mask)[0]
        history_repr = history_embedding[:, 0, :]
        history_repr = history_repr.view(batch_size, his_length, -1)

        if self.apply_reduce_dim:
            candidate_repr = self.reduce_embed_dim(candidate_repr)
            candidate_repr = self.dropout(candidate_repr)
            history_repr = self.reduce_embed_dim(history_repr)
            history_repr = self.dropout(history_repr)

        # Multi-interest user modeling
        if self.use_category_bias:
            history_category_embed = self.category_embedding(history_category_encoding)
            history_category_embed = self.category_dropout(history_category_embed)
            candidate_category_embed = self.category_embedding(candidate_category_encoding)
            candidate_category_embed = self.category_dropout(candidate_category_embed)
            category_bias = pairwise_cosine_similarity(history_category_embed, candidate_category_embed)

            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=history_mask, bias=category_bias)
        else:
            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=history_mask, bias=None)
        multi_user_interest = self.dropout(multi_user_interest)

        # Click predictor
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if self.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif self.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif self.score_type == 'weighted':
            matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
                                                     value=matching_scores)
        else:
            raise ValueError('Invalid method of aggregating matching score')

        return multi_user_interest, matching_scores

    @property
    def news_embed_dim(self):
        return self._embed_dim


class PolyAttention(nn.Module):
    r"""
    Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
    """
    def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
        r"""
        Initialization

        Args:
            in_embed_dim: The number of expected features in the input ``embeddings``
            num_context_codes: The number of attention vectors ``K``
            context_code_dim: The number of features in a context code
        """
        super().__init__()
        self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings: Tensor, attn_mask: Tensor, bias: Tensor = None):
        r"""
        Forward propagation

        Args:
            embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``
            attn_mask: tensor of shape ``(batch_size, his_length)``
            bias: tensor of shape ``(batch_size, his_length, num_candidates)``

        Returns:
            A tensor of shape ``(batch_size, num_context_codes, embed_dim)``
        """
        proj = torch.tanh(self.linear(embeddings))
        if bias is None:
            weights = torch.matmul(proj, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(proj, self.context_codes.T) + bias
        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = torch_f.softmax(weights, dim=2)
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr


class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        r"""
        Initialization

        Args:
            embed_dim: The number of features in query and key vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
            key: tensor of shape ``(batch_size, num_candidates, embed_dim)``
            value: tensor of shape ``(batch_size, num_candidates, num_context_codes)``

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        proj = torch_f.gelu(self.linear(query))
        weights = torch_f.softmax(torch.matmul(key, proj.permute(0, 2, 1)), dim=2)
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs
