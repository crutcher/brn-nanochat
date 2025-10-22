//! # Norm Extensions
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::DType::F32;

/// Apply root-mean-square norm.
pub fn rms_norm<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let eps: f32 = 1e-7;
    let dtype = x.dtype();

    let rms = (x.clone().cast(F32).square().mean_dim(-1) + eps).sqrt();

    x / rms.cast(dtype)
}
