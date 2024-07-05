from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, KVCache, create_additive_causal_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "blenderbot"
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    add_bias_logits: bool = False
    add_final_layer_norm: bool = True
    attention_dropout: float = 0.0
    bos_token_id: int = 1
    classif_dropout:float = 0.0
    d_model: int = 2560
    decoder_attention_heads: int = 32
    decoder_ffn_dim: int = 10240
    decoder_layerdrop: float = 0.0
    decoder_layers: int = 24
    dropout: float = 0.1
    encoder_attention_heads: int = 32
    encoder_ffn_dim: int = 10240
    encoder_layerdrop: float = 0.0
    encoder_layers: int = 2
    eos_token_id: int = 2
    extra_layer_norm: bool = False
    extra_pos_embeddings: int = 0
    force_bos_token_to_be_generated: bool = False
    # id2label:int = {
    #     "0:int = "LABEL_0",
    #     "1:int = "LABEL_1",
    #     "2:int = "LABEL_2"
    # }
    init_std: float = 0.02
    is_encoder_decoder: bool = True
    # label2id: int = {
    #     "LABEL_0:int = 0,
    #     "LABEL_1:int = 1,
    #     "LABEL_2:int = 2
    # }
    length_penalty: float = 0.65
    max_length: int = 60
    max_position_embeddings: int = 128
    min_length: int = 20
    no_repeat_ngram_size: int = 3
    normalize_before: bool = True
    normalize_embedding: bool = False
    num_beams: int = 10
    num_hidden_layers: int = 2
    pad_token_id: int = 0
    scale_embedding: bool = True
    static_position_embeddings: bool = False
    unk_token_id: int = 3
    layernorm_variant: str = "prelayernorm"
    vocab_size: int = 8008


class BlenderbotLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def __call__(self, input_ids_shape: mx.array, cache: int = 0):
        B, L = input_ids_shape[:2]
        positions = mx.arange(cache, cache + L, dtype=input_ids_shape)
        return super().__call__(positions)
    
class BlenderbotScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def __call__(self, input_ids: mx.array):
        return super().__call__(input_ids) * self.embed_scale
    

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()