//! # GPT Module

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::nn::activation::ActivationConfig;
use burn::prelude::{Backend, Int};
use burn::Tensor;
use crate::burn_ext::norm::rms_norm;
use crate::model::block::{GPTBlock, GPTBlockConfig};
use crate::burn_ext::nn::embedding::rotary::{RotaryEmbedding, RotaryEmbeddingConfig, RotaryEmbeddingMeta};
use crate::model::csa::CausalSelfAttentionConfig;
use crate::model::mlp::MLPConfig;

/// Common meta for [`GPT`] and [`GPTConfig`].
pub trait GPTMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// High-level GPT Config.
#[derive(Config, Debug)]
pub struct GPTConfig {
    /// Sequence Length.
    #[config(default = "1024")]
    pub sequence_len: usize,

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
}

impl GPTConfig {
    /// Initialize a [`GPT`].
    pub fn init<B: Backend>(self, device: &B::Device) -> GPT<B> {
        self.into_structure().init(device)
    }

    /// Convert this config into a [`GPTStructureConfig`].
    pub fn into_structure(self) -> GPTStructureConfig {
        let wte = EmbeddingConfig::new(self.vocab_size, self.n_embed);

        let block_config = self.block_config();

        let h = (0..self.n_layer)
            .map(|_| block_config.clone())
            .collect();

        let lm_head = LinearConfig::new(self.n_embed, self.vocab_size);

        let rotary_seq_len = self.sequence_len * self.rotary_sequence_factor;
        let rotary_embedding = RotaryEmbeddingConfig::new(rotary_seq_len, self.head_dim());

        GPTStructureConfig { wte, h, lm_head, rotary_embedding }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embed / self.n_head
    }

    /// Build the [`GPTBlockConfig`] for this config.
    pub fn block_config(
        &self,
    ) -> GPTBlockConfig {
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
    wte: EmbeddingConfig,
    h: Vec<GPTBlockConfig>,
    lm_head: LinearConfig,
    rotary_embedding: RotaryEmbeddingConfig,
}

impl GPTStructureConfig {
    /// Initialize a [`GPT`].
    pub fn init<B: Backend>(self, device: &B::Device) -> GPT<B> {
        let wte = self.wte.init(device);
        let h = self.h.into_iter().enumerate()
            .map(|(layer_idx, c)| c.init(layer_idx, device))
            .collect();
        let lm_head = self.lm_head.init(device);
        let rotary_embedding = self.rotary_embedding.init(device);

        GPT { wte, h, lm_head, rotary_embedding }
    }
}

/// GPT Module
#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    wte: Embedding<B>,
    h: Vec<GPTBlock<B>>,
    lm_head: Linear<B>,
    rotary_embedding: RotaryEmbedding<B>,
}

impl<B: Backend> GPTMeta for GPT<B> {
    fn n_embed(&self) -> usize {
        self.wte.weight.dims()[0]
    }
}

impl<B: Backend> GPT<B> {
    /// Forward Pass.
    pub fn forward(&self,
       idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, t] = unpack_shape_contract!(
            ["B", "T"],
            &idx.dims()
        );
        assert!(t <= self.rotary_embedding.seq_len(),
            "Sequence length grew beyond the rotary embeddings cache: {t} > {}",
            self.rotary_embedding.seq_len()
        );

        let mut x = self.wte.forward(idx);
        x = rms_norm(x);
        for block in &self.h {
            x = block.forward(x, &self.rotary_embedding);
        }
        x = rms_norm(x);
        let softcap = 15.0;

        let logits = self.lm_head.forward(x);
        let logits: Tensor<B, 3> = softcap * (logits / softcap).tanh();

        assert_shape_contract_periodically!(
            ["B", "T", "D"],
            &logits.dims(),
            &[("B", b), ("T", t), ("D", self.n_embed())]
        );
        logits
    }
}