//! # GPT Block

use crate::burn_ext::nn::embedding::rotary::RotaryEmbedding;
use crate::burn_ext::norm::rms_norm;
use crate::model::csa::{CausalSelfAttention, CausalSelfAttentionConfig, CausalSelfAttentionMeta};
use crate::model::mlp::{MLP, MLPConfig, MLPMeta};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;

/// Common meta for [`GPTBlock`] and [`GPTBlockConfig`].
pub trait GPTBlockMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`GPTBlock`].
#[derive(Config, Debug)]
pub struct GPTBlockConfig {
    /// Causal Self-Attention Config.
    pub attn: CausalSelfAttentionConfig,

    /// MLP Config.
    pub mlp: MLPConfig,
}

impl GPTBlockMeta for GPTBlockConfig {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl GPTBlockConfig {
    /// Initialize a [`GPTBlock`].
    pub fn init<B: Backend>(
        self,
        layer_index: usize,
        device: &B::Device,
    ) -> GPTBlock<B> {
        assert_eq!(self.attn.n_embed(), self.mlp.n_embed());
        GPTBlock {
            attn: self.attn.init(layer_index, device),
            mlp: self.mlp.init(device),
        }
    }
}

/// GPT Block
#[derive(Module, Debug)]
pub struct GPTBlock<B: Backend> {
    pub attn: CausalSelfAttention<B>,
    pub mlp: MLP<B>,
}

impl<B: Backend> GPTBlockMeta for GPTBlock<B> {
    fn n_embed(&self) -> usize {
        self.attn.n_embed()
    }
}

impl<B: Backend> GPTBlock<B> {
    /// Forward Pass.
    ///
    /// # Arguments
    /// - `input`: a ``[B, T, D]`` input.
    /// - `rotary_embedding`: a ``[1, T, 1, D/2]`` embedding.
    ///
    /// # Returns
    /// - the ``[B, T, D]`` block output.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        rotary_embedding: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let x = rms_norm(input);
        let x = self.attn.forward(x, rotary_embedding);
        let x = rms_norm(x);
        self.mlp.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn_ext::nn::embedding::rotary::RotaryEmbeddingConfig;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::{Cuda, Wgpu};
    use burn::tensor::Distribution;

    #[test]
    fn test_gpt_block_config() {
        type B = Wgpu;
        let device = Default::default();

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;

        let config = GPTBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MLPConfig::new(n_embed),
        );
        assert_eq!(config.n_embed(), n_embed);
        assert_eq!(config.attn.n_embed(), n_embed);
        assert_eq!(config.attn.n_head(), n_head);
        assert_eq!(config.attn.n_kv_head(), n_kv_head);

        assert_eq!(config.mlp.n_embed(), n_embed);

        let layer_index = 12;
        let block: GPTBlock<B> = config.init(layer_index, &device);

        assert_eq!(block.n_embed(), n_embed);
    }

    #[test]
    fn test_gpt_block_forward() {
        type B = Cuda;
        let device = Default::default();

        let batch = 2;
        let seq_len = 10;

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;
        let layer_index = 12;

        let config = GPTBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MLPConfig::new(n_embed),
        );

        let block: GPTBlock<B> = config.init(layer_index, &device);

        let input = Tensor::random([batch, seq_len, n_embed], Distribution::Default, &device);

        let rotary_embedding =
            RotaryEmbeddingConfig::new(seq_len, block.attn.head_dim()).init(&device);

        let output = block.forward(input.clone(), &rotary_embedding);
        assert_shape_contract!(
            ["B", "T", "D"],
            &output.dims(),
            &[("B", batch), ("T", seq_len), ("D", n_embed)]
        );
    }
}
