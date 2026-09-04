import einops
from jaxtyping import Bool, Float, Int
from torch import Tensor

def my_run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    
    lineared = einops.einsum( in_features, weights , "... d_in , d_out d_in ->... d_out")
    return lineared
