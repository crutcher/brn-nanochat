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
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rotary_embedding: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let x = rms_norm(x);
        let x = self.attn.forward(x, rotary_embedding);
        let x = rms_norm(x);
        self.mlp.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    fn test_gpt_block_config() {
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
    }

    #[test]
    fn test_gpt_block() {
        type B = Wgpu;
        let device = Default::default();

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;

        let config = GPTBlockConfig::new(
            CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed),
            MLPConfig::new(n_embed),
        );

        let layer_index = 12;
        let block: GPTBlock<B> = config.init(layer_index, &device);

        assert_eq!(block.n_embed(), n_embed);
    }
}
