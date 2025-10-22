//! # Rotary Embedding
use burn::Tensor;
use burn::prelude::{Backend, s};

/// Apply rotary embedding.
pub fn apply_rotary_embedding<B: Backend>(
    x: Tensor<B, 4>,
    cos_sin: (f32, f32),
) -> Tensor<B, 4> {
    let dtype = x.dtype();
    let (cos, sin) = cos_sin;

    let d = x.dims()[3];
    let x1 = x.clone().slice_dim(3, s![..d]);
    let x2 = x.clone().slice_dim(3, s![d..]);

    let y1 = x1.clone() * cos + x2.clone() * sin;
    let y2 = x1 * (-sin) + x2 * cos;

    Tensor::cat(vec![y1, y2], 3).cast(dtype)
}
