//! # Rotary Embedding

use crate::burn_ext::tensor::outer;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, s};
use burn::tensor::DType;
use std::ops::Range;

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
    ///
    /// This must be even.
    pub head_dim: usize,

    /// Base.
    #[config(default = 10000)]
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
        assert!(
            self.head_dim.is_multiple_of(2),
            "Head dimension must be even: {}",
            self.head_dim
        );

        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let base = self.base;

        let channel_range: Tensor<B, 1> =
            Tensor::arange_step(0..head_dim as i64, 2, device).float();
        let base: Tensor<B, 1> = Tensor::from_data([base as f32], device);
        let inv_freq: Tensor<B, 1> = base.powf(-channel_range / head_dim as f32);

        let t: Tensor<B, 1> = Tensor::arange(0..seq_len as i64, device).float();

        let freqs = outer(t, inv_freq);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        // TODO: possibly down-cast to the smallest available dtype.

        let cos = cos
            .set_require_grad(false)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim(0);
        let sin = sin
            .set_require_grad(false)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim(0);

        // [1, T, 1, D/2]

        RotaryEmbedding { head_dim, cos, sin }
    }
}

/// Rotary Embedding Module
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Head Dimension, D
    pub head_dim: usize,

    /// a ``[1, T, 1, D/2]`` tensor.
    pub cos: Tensor<B, 4>,

    /// a ``[1, T, 1, D/2]`` tensor.
    pub sin: Tensor<B, 4>,
}

impl<B: Backend> RotaryEmbeddingMeta for RotaryEmbedding<B> {
    fn seq_len(&self) -> usize {
        self.cos.dims()[1]
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Cast the embedding to a different dtype.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            cos: self.cos.cast(dtype),
            sin: self.sin.cast(dtype),
            ..self
        }
    }

    /// Clip the embedding to cover only the given range.
    ///
    /// # Arguments
    /// - `range`: the ``start..end`` range to cover.
    ///
    /// # Returns
    /// - a clipped [`RotaryEmbedding`].
    pub fn clip_range(
        &self,
        range: Range<usize>,
    ) -> Self {
        Self {
            head_dim: self.head_dim,
            cos: self.cos.clone().slice_dim(1, range.clone()),
            sin: self.sin.clone().slice_dim(1, range),
        }
    }

    /// Apply the rotary embedding to the input.
    ///
    /// # Arguments
    /// - `input`: a ``[B, T, H, D]`` tensor.
    ///
    /// # Returns
    /// - a ``[B, T, H, D]`` tensor.
    pub fn apply(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [b, h] = unpack_shape_contract!(
            ["B", "T", "H", "D"],
            &input.dims(),
            &["B", "H"],
            &[("T", self.seq_len()), ("D", self.head_dim())]
        );

        let pivot = self.head_dim() / 2;
        let x1 = input.clone().slice_dim(3, s![..pivot]);
        let x2 = input.clone().slice_dim(3, s![pivot..]);

        let y1 = x1.clone() * self.cos.clone() + x2.clone() * self.sin.clone();
        let y2 = x1 * (-self.sin.clone()) + x2 * self.cos.clone();

        let output = Tensor::cat(vec![y1, y2], 3);

        assert_shape_contract_periodically!(
            ["B", "T", "H", "D"],
            &output.dims(),
            &[
                ("B", b),
                ("T", self.seq_len()),
                ("H", h),
                ("D", self.head_dim())
            ]
        );

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Cuda;

    #[test]
    fn test_clip_range() {
        type B = Cuda;
        let device = Default::default();

        let config = RotaryEmbeddingConfig::new(1024, 64);
        let re: RotaryEmbedding<B> = config.init(&device);
        assert_eq!(re.seq_len(), 1024);
        assert_eq!(re.head_dim(), 64);

        let clip_re = re.clip_range(10..20);
        assert_eq!(clip_re.seq_len(), 10);
        clip_re
            .sin
            .clone()
            .to_data()
            .assert_eq(&re.sin.clone().slice_dim(1, 10..20).to_data(), true);
        clip_re
            .cos
            .clone()
            .to_data()
            .assert_eq(&re.cos.clone().slice_dim(1, 10..20).to_data(), true);
    }

    #[test]
    fn test_rotary_embedding() {
        type B = Cuda;
        let device = Default::default();

        let config = RotaryEmbeddingConfig::new(1024, 64);
        assert_eq!(config.seq_len(), 1024);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.base, 10000);

        let re: RotaryEmbedding<B> = config.init(&device);
        assert_eq!(re.seq_len(), 1024);
        assert_eq!(re.head_dim(), 64);
    }
}
