import torch
from jaxtyping import Float, Int
from torch import Tensor
import einops
def my_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    device = in_query_or_key.device
    i = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
    freq = theta ** (-i/d_k)
    angles = einops.einsum(
        token_positions.to(torch.float32), freq,
        "... sequence_length, half_d_k -> ... sequence_length half_d_k",
    )
    #两两分组
    angles_paired = einops.rearrange(
        in_query_or_key,
        "... sequence_length (half_d_k two) -> ... sequence_length half_d_k two",
        two = 2,
    )
    angles_complex = torch.view_as_complex(angles_paired)

    #构建单位旋转复数
    rot_complex = torch.polar(torch.ones_like(angles), angles)
    #旋转计算与复数还原
    x_out_complex = angles_complex * rot_complex
    x_out_paired = torch.view_as_real(x_out_complex)
    x_out = einops.rearrange(
        x_out_paired,
        "... sequence_length half_d_k two -> ... sequence_length (half_d_k two)",
        two=2,
    )
    return x_out