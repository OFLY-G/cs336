from jaxtyping import Bool, Float, Int
from torch import Tensor
import einops
import math
import torch

def my_scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None=None,
) -> Float[Tensor, "... queries d_v"]:

    d_k = Q.shape[-1]
    scaled_scores = (
            einops.einsum(
            Q ,K, "... queries d_k, ... keys d_k -> ... queries keys"
            )
            / math.sqrt(d_k)
        )
    
    if mask==None:
        masked_scores = scaled_scores
    else:
        masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))

    attention_weight = torch.softmax(masked_scores, dim= -1)

    output = einops.einsum(
        attention_weight, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )
    return output