from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel


class Miner(ABC, RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, apply_reduce_dim: bool, reduce_dim: int, use_category_bias: bool,
                 num_category: int, category_embed_dim: int, category_pad_token_id: int, num_context_codes: int,
                 context_code_dim: int, score_type: str, dropout: float):
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
            self.category_weight = nn.Parameter(torch.randn(1))

        self.poly_attn = PolyAttention(in_embed_dim=self._news_embed_dim, num_context_codes=num_context_codes,
                                       context_code_dim=context_code_dim)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self._news_embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, history_encoding: torch.tensor, history_attn_mask: torch.tensor,
                history_category_encoding: torch.tensor, history_mask: torch.tensor, candidate_encoding: torch.tensor,
                candidate_attn_mask: torch.tensor, candidate_category_encoding: torch.tensor):
        """
        Forward propagation
        :param history_encoding:
        :param history_attn_mask:
        :param history_category_encoding:
        :param history_mask:
        :param candidate_encoding:
        :param candidate_attn_mask:
        :param candidate_category_encoding:
        :type history_encoding: torch.tensor, shape [batch_size, his_length, seq_length]
        :type history_attn_mask: torch.tensor, shape [batch_size, his_length, seq_length]
        :type history_category_encoding: torch.tensor, shape [batch_size, his_length]
        :type candidate_encoding: torch.tensor, shape [batch_size, num_candidates, seq_length]
        :type candidate_attn_mask: torch.tensor, shape [batch_size, num_candidates, seq_length]
        :type candidate_category_encoding: torch.tensor, shape [batch_size, num_candidates]
        :return:
        """
        batch_size = history_encoding.shape[0]
        his_length = history_encoding.shape[1]
        num_candidates = candidate_encoding.shape[1]

        # Representation of candidate news
        candidate_encoding = candidate_encoding.view(batch_size * num_candidates, -1)
        candidate_attn_mask = candidate_attn_mask.view(batch_size * num_candidates, -1)
        candidate_embedding = self.roberta(input_ids=candidate_encoding, attention_mask=candidate_attn_mask)
        candidate_repr = candidate_embedding[:, 0, :]
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)

        # Representation of history clicked news
        history_encoding = history_encoding.view(batch_size * his_length, -1)
        history_attn_mask = history_attn_mask.view(batch_size * his_length, -1)
        history_embedding = self.roberta(input_ids=history_encoding, attention_mask=history_attn_mask)[0]
        history_repr = history_embedding[:, 0, :]
        history_repr = history_repr.view(batch_size, his_length, -1)

        # Multi-interest user modeling
        if self.use_category_bias:
            history_category_embed = self.category_embedding(history_category_encoding)
            history_category_embed = nn.Dropout(history_category_embed)
            candidate_category_embed = self.category_embedding(candidate_category_encoding)
            candidate_category_embed = nn.Dropout(candidate_category_embed)
            category_bias =
        multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=history_mask, bias=None)

        # Click predictor
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if self._score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif self._score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif self._score_type == 'weighted':
            matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
                                                     value=matching_scores)
        else:
            raise ValueError('Invalid method of aggregating matching score')

        return multi_user_interest, matching_scores

    @property
    def news_embed_dim(self):
        return self._embed_dim


class PolyAttention(nn.Module):
    def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings: torch.tensor, attn_mask: torch.tensor, bias: torch.tensor):
        """
        Forward propagation
        :param embeddings: The sequence of historical user behaviors' representation
        :param attn_mask: Mask to avoid focusing attention on padding behavior indices.
        :param bias:
        :type embeddings: torch.tensor, shape [batch_size, his_length, embed_dim]
        :type attn_mask: torch.tensor, shape [batch_size, his_length]
        :return:
        :rtype: torch.tensor, shape [batch_size, num_context_codes, embed_dim]
        """
        proj = torch.tanh(self.linear(embeddings))
        if bias is None:
            weights = torch.matmul(proj, self.context_codes.T).permute(0, 2, 1)
        else:
            weights = torch.matmul(proj, self.context_codes.T).permute(0, 2, 1) + bias
        weights = weights.masked_fill_(~attn_mask, 1e-30)
        weights = torch_f.softmax(weights, dim=2)
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr


class TargetAwareAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor):
        """
        Forward propagation
        :param query:
        :param key:
        :param value:
        :type query: torch.tensor, shape [batch_size, num_context_codes, embed_dim]
        :type key: torch.tensor, shape [batch_size, num_candidates, embed_dim]
        :type value: torch.tensor, shape [batch_size, num_candidates, num_context_codes]
        :return:
        :rtype: torch.tensor, shape [batch_size, num_candidates]
        """
        proj = torch_f.gelu(self.linear(query))
        weights = torch_f.softmax(torch.matmul(key, proj.permute(0, 2, 1)))
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs


def pairwise_cosine_similarity(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Calculates the pairwise cosine similarity matrix
    :param x:
    :param y:
    :type x: torch.tensor, shape [batch_size, M, d]
    :type y: torch.tensor, shape [batch_size, N, d]
    :return:
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    result = torch.matmul(x_norm, y_norm.permute(0, 2, 1))

    return result







