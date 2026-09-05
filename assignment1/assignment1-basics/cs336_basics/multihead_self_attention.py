from torch import Tensor
import einops
import torch
from jaxtyping import Bool, Float, Int
from cs336_basics.scaled_dot_product_attention import my_scaled_dot_product_attention 
 
def my_run_multihead_self_attention(
   d_model:int,
   num_heads: int,
   q_proj_weight: Float[Tensor, " d_model d_model"],
   k_proj_weight: Float[Tensor, " d_model d_model"],
   v_proj_weight: Float[Tensor, " d_model d_model"],
   o_proj_weight: Float[Tensor, " d_model d_model"],
   in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
   d_head= d_model // num_heads
   seq_len=in_features.shape[-2]
   Q = einops.einsum(
       q_proj_weight, in_features, 
       "d_out d_in, ... sequence_length d_in -> ... sequence_length d_out"
   )
   K = einops.einsum(
       k_proj_weight, in_features, 
       "d_out d_in, ... sequence_length d_in -> ... sequence_length d_out"
   )
   V = einops.einsum(
       v_proj_weight, in_features, 
       "d_out d_in, ... sequence_length d_in -> ... sequence_length d_out"
   )
   #拆分多头
   Q = einops.rearrange(
       Q,
       "... sequence_length (h d_head) -> ... h sequence_length d_head",
       h=num_heads
   )
   K = einops.rearrange(
       K,
       "... sequence_length (h d_head) -> ... h sequence_length d_head",
       h=num_heads
   )
   V = einops.rearrange(
       V,
       "... sequence_length (h d_head) -> ... h sequence_length d_head",
       h=num_heads
   )
   #mask
   causal_mask = torch.tril(
       torch.ones(
           (seq_len, seq_len), dtype = torch.bool, device = in_features.device
       )
   )
   #注意力
   attn_out = my_scaled_dot_product_attention(Q, K, V, mask=causal_mask)
   #合并多头
   output = einops.rearrange(
       attn_out,
       "... h sequence_length d_head -> ... sequence_length (h d_head)"
   )
   #投影
   output = einops.einsum(
       output, o_proj_weight,
       "... sequence_length d_in, d_out d_in -> ... sequence_length d_out"
   )
   return output

#rope版本
def my_run_multihead_self_attention_with_rope(
   d_model:int,
   num_heads: int,
   max_seq_len: int,
   theta: float,
   q_proj_weight: Float[Tensor, " d_model d_model"],
   k_proj_weight: Float[Tensor, " d_model d_model"],
   v_proj_weight: Float[Tensor, " d_model d_model"],
   o_proj_weight: Float[Tensor, " d_model d_model"],
   in_features: Float[Tensor, " ... sequence_length d_model"],
   token_positions: Int[Tensor, "... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
   d_head= d_model // num_heads
   seq_len=in_features.shape[-2]
   Q = einops.einsum(
       q_proj_weight, in_features, 
       "d_out d_in, ... sequence_length d_in -> ... sequence_length d_out"
   )
   K = einops.einsum(
       k_proj_weight, in_features, 
       "d_out d_in, ... sequence_length d_in -> ... sequence_length d_out"
   )
   V = einops.einsum(
       v_proj_weight, in_features, 
       "d_out d_in, ... sequence_length d_in -> ... sequence_length d_out"
   )
   #拆分多头
   Q = einops.rearrange(
       Q,
       "... sequence_length (h d_head) -> ... h sequence_length d_head",
       h=num_heads
   )
   K = einops.rearrange(
       K,
       "... sequence_length (h d_head) -> ... h sequence_length d_head",
       h=num_heads
   )
   V = einops.rearrange(
       V,
       "... sequence_length (h d_head) -> ... h sequence_length d_head",
       h=num_heads
   )
   #mask
   causal_mask = torch.tril(
       torch.ones(
           (seq_len, seq_len), dtype = torch.bool, device = in_features.device
       )
   )
   #注意力
   attn_out = my_scaled_dot_product_attention(Q, K, V, mask=causal_mask)
   #合并多头
   output = einops.rearrange(
       attn_out,
       "... h sequence_length d_head -> ... sequence_length (h d_head)"
   )
   #投影
   output = einops.einsum(
       output, o_proj_weight,
       "... sequence_length d_in, d_out d_in -> ... sequence_length d_out"
   )
   return output