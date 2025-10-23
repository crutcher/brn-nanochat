//! # Rotary Embedding

use burn::config::Config;
use burn::module::{Module};
use burn::prelude::Backend;
use burn::Tensor;
use crate::burn_ext::nn::functional::embedding::rotary::apply_rotary_embedding;
use crate::burn_ext::tensor::outer;

/// Common meta for [`RotaryEmbedding`] and [`RotaryEmbeddingConfig`].
pub trait RotaryEmbeddingMeta {
    /// Return the sequence length.
    fn seq_len(&self) -> usize;

    /// Return the head dimension.
    fn head_dim(&self) -> usize;
}

/// Config for [`RotaryEmbedding`].
#[derive(Config, Debug)]
pub struct RotaryEmbeddingConfig {
    /// Sequence Length.
    pub seq_len: usize,

    /// Head Dimension.
    pub head_dim: usize,

    /// Base.
    #[config(default= 10000)]
    pub base: usize,
}

impl RotaryEmbeddingMeta for RotaryEmbeddingConfig {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl RotaryEmbeddingConfig {
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> RotaryEmbedding<B> {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let base = self.base;
        let channel_range: Tensor<B, 1> = Tensor::arange_step(0..head_dim as i64, 2, device).float();
        let base = channel_range.full_like(base as f32);
        let inv_freq: Tensor<B, 1> = base.powf(-channel_range / head_dim as f32);

        let t: Tensor<B, 1> = Tensor::arange(0..seq_len as i64, device).float();

        let freqs = outer(t, inv_freq);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        // TODO: possibly down-cast to the smallest available dtype.

        let cos = cos.set_require_grad(false).unsqueeze_dim::<3>(1).unsqueeze_dim(0);
        let sin = sin.set_require_grad(false).unsqueeze_dim::<3>(1).unsqueeze_dim(0);

        RotaryEmbedding { cos, sin }
    }
}

/// Rotary Embedding Module
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// a ``[1, T, 1, D]`` tensor.
    pub cos: Tensor<B, 4>,

    /// a ``[1, T, 1, D]`` tensor.
    pub sin: Tensor<B, 4>,
}

impl<B: Backend> RotaryEmbeddingMeta for RotaryEmbedding<B> {
    fn seq_len(&self) -> usize {
        self.cos.dims()[1]
    }

    fn head_dim(&self) -> usize {
        self.cos.dims()[3]
    }
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn apply(
        &self,
        x: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        apply_rotary_embedding(x, &self)
    }
}