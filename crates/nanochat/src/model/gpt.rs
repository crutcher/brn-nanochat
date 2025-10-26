//! # GPT Module

use crate::burn_ext::nn::embedding::rotary::{
    RotaryEmbedding, RotaryEmbeddingConfig, RotaryEmbeddingMeta,
};
use crate::burn_ext::norm::rms_norm;
use crate::model::block::{GPTBlock, GPTBlockConfig};
use crate::model::csa::CausalSelfAttentionConfig;
use crate::model::mlp::MLPConfig;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::activation::ActivationConfig;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Int};

/// Common meta for [`GPT`] and [`GPTConfig`].
pub trait GPTMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;

    /// Return the size of the rotary embedding cache.
    fn seq_len(&self) -> usize;
}

/// High-level GPT Config.
#[derive(Config, Debug)]
pub struct GPTConfig {
    /// Sequence Length.
    #[config(default = "1024")]
    pub seq_len: usize,

    /// Vocabulary Size.
    #[config(default = "50304")]
    pub vocab_size: usize,

    /// Number of Blocks.
    #[config(default = "12")]
    pub n_layer: usize,

    /// Number of Query Heads.
    #[config(default = "6")]
    pub n_head: usize,

    /// Number of KV Heads.
    #[config(default = "6")]
    pub n_kv_head: usize,

    /// Embedding Size.
    #[config(default = "768")]
    pub n_embed: usize,

    /// Softcap for the logits.
    #[config(default = "15.0")]
    pub softcap: f64,

    /// MLP Expansion Factor.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// MLP Activation Config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,

    /// Over-compute factor for rotary embeddings.
    #[config(default = "10")]
    pub rotary_sequence_factor: usize,
}

impl GPTMeta for GPTConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }
}

impl GPTConfig {
    /// Initialize a [`GPT`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> GPT<B> {
        self.into_structure().init(device)
    }

    /// Convert this config into a [`GPTStructureConfig`].
    pub fn into_structure(self) -> GPTStructureConfig {
        let wte = EmbeddingConfig::new(self.vocab_size, self.n_embed);

        let block_config = self.block_config();

        let h = (0..self.n_layer).map(|_| block_config.clone()).collect();

        let lm_head = LinearConfig::new(self.n_embed, self.vocab_size);

        let rotary_seq_len = self.seq_len * self.rotary_sequence_factor;
        let rotary_embedding = RotaryEmbeddingConfig::new(rotary_seq_len, self.head_dim());

        GPTStructureConfig::new(wte, h, lm_head, rotary_embedding).with_softcap(self.softcap)
    }

    pub fn head_dim(&self) -> usize {
        self.n_embed / self.n_head
    }

    /// Build the [`GPTBlockConfig`] for this config.
    pub fn block_config(&self) -> GPTBlockConfig {
        GPTBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_head, self.n_kv_head, self.n_embed),
            MLPConfig::new(self.n_embed)
                .with_expansion_factor(self.expansion_factor)
                .with_activation(self.activation.clone()),
        )
    }
}

/// Low-level GPT Structure Config.
///
/// This config has a lot of duplicate information.
#[derive(Config, Debug)]
pub struct GPTStructureConfig {
    pub wte: EmbeddingConfig,
    pub h: Vec<GPTBlockConfig>,
    pub lm_head: LinearConfig,
    pub rotary_embedding: RotaryEmbeddingConfig,

    /// Softcap for the logits.
    #[config(default = "15.0")]
    pub softcap: f64,
}

impl GPTMeta for GPTStructureConfig {
    fn n_embed(&self) -> usize {
        self.wte.n_embedding
    }

    fn seq_len(&self) -> usize {
        self.rotary_embedding.seq_len()
    }
}

impl GPTStructureConfig {
    /// Initialize a [`GPT`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> GPT<B> {
        let wte = self.wte.init(device);
        let h = self
            .h
            .into_iter()
            .enumerate()
            .map(|(layer_idx, c)| c.init(layer_idx, device))
            .collect();
        let lm_head = self.lm_head.init(device);
        let re = self.rotary_embedding.init(device);

        GPT {
            wte,
            h,
            lm_head,
            re,
            softcap: self.softcap,
        }
    }
}

/// GPT Module
#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    wte: Embedding<B>,
    h: Vec<GPTBlock<B>>,
    lm_head: Linear<B>,
    re: RotaryEmbedding<B>,
    softcap: f64,
}

impl<B: Backend> GPTMeta for GPT<B> {
    fn n_embed(&self) -> usize {
        self.wte.weight.dims()[0]
    }

    fn seq_len(&self) -> usize {
        self.re.seq_len()
    }
}

impl<B: Backend> GPT<B> {
    /// Forward Pass.
    pub fn forward(
        &self,
        idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, t] = unpack_shape_contract!(["B", "T"], &idx.dims());
        assert!(
            t <= self.re.seq_len(),
            "Sequence length grew beyond the rotary embeddings cache: {t} > {}",
            self.re.seq_len()
        );

        let mut x = self.wte.forward(idx);

        // Note: The reference nanochat has a norm here,
        // but the block has the same norm as the first operation.
        // x = rms_norm(x);

        for block in &self.h {
            x = block.forward(x, &self.re);
        }

        x = rms_norm(x);

        let logits = self
            .lm_head
            .forward(x)
            .div_scalar(self.softcap)
            .tanh()
            .mul_scalar(self.softcap);

        assert_shape_contract_periodically!(
            ["B", "T", "D"],
            &logits.dims(),
            &[("B", b), ("T", t), ("D", self.n_embed())]
        );
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config() {
        let cfg = GPTConfig::new();
        assert_eq!(cfg.seq_len, 1024);
        assert_eq!(cfg.vocab_size, 50304);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 6);
        assert_eq!(cfg.n_kv_head, 6);
        assert_eq!(cfg.n_embed, 768);
        assert_eq!(cfg.expansion_factor, 4);

        assert_eq!(cfg.n_embed(), 768);
    }
}
