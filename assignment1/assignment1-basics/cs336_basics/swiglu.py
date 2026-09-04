from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum
import torch

def my_swiglu(
    w1_weight: Float[Tensor, "d_ff d_model"],
    w2_weight: Float[Tensor, "d_model d_ff"],
    w3_weight: Float[Tensor, "d_ff d_model"],
    in_features: Float[Tensor, "... d_model"]
) -> Float[Tensor, "... d_model"]:
    beforeactivate = einsum(in_features, w1_weight , "... d_model, d_ff d_model -> ... d_ff")
    SiLU = beforeactivate * torch.sigmoid(beforeactivate) 
    gate = einsum(in_features, w3_weight , "... d_model, d_ff d_model -> ... d_ff")
    
    glu = einsum(SiLU , gate, "... d_ff , ... d_ff-> ... d_ff")
    output = einsum(glu, w2_weight, "... d_ff, d_model d_ff -> ... d_model")
    return output