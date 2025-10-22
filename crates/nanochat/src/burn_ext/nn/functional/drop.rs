//! # Drop Extensions

use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::Distribution;

/// Applies functional dropout on the input tensor.
///
/// Always applies, does check for training mode.
pub fn dropout<B: Backend, const D: usize>(
    prob: f64,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    if prob == 0.0 {
        return input;
    }

    let prob_keep = 1.0 - prob;
    let random = input.random_like(Distribution::Bernoulli(prob_keep));
    let x = input * random;

    x * (1.0 / prob_keep)
}
