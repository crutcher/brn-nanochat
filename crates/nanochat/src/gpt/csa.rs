//! # Causal Self-Attention

use crate::burn_ext::nn::functional::attention;
use crate::burn_ext::nn::functional::attention::ScaledDotProductAttentionConfig;
use crate::burn_ext::nn::functional::embedding::rotary;
use crate::burn_ext::norm;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;

/// Common meta for [`CausalSelfAttention`] and [`CausalSelfAttentionConfig`].
pub trait CausalSelfAttentionMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;

    /// Return the number of heads.
    fn n_head(&self) -> usize;

    /// Return the number of KV heads.
    fn n_kv_head(&self) -> usize;

    /// Return the size of each head.
    fn head_dim(&self) -> usize {
        self.n_embed() / self.n_head()
    }

    /// Group Query Attention.
    fn gqa_enabled(&self) -> bool {
        self.n_head() != self.n_kv_head()
    }
}

/// Config for [`CausalSelfAttention`].
#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    /// Number of Heads.
    pub n_head: usize,

    /// Number of KV Heads.
    pub n_kv_head: usize,

    /// Embedding Size.
    pub n_embed: usize,
}

impl CausalSelfAttentionMeta for CausalSelfAttentionConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn n_head(&self) -> usize {
        self.n_head
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }
}

impl CausalSelfAttentionConfig {
    /// Return the size of each head.
    pub fn head_dim(&self) -> usize {
        self.n_embed / self.n_head
    }

    /// Validate the config.
    pub fn validate(&self) {
        assert!(self.n_embed.is_multiple_of(self.n_head));
        assert!(self.n_embed.is_multiple_of(self.n_kv_head));
    }
}

impl CausalSelfAttentionConfig {
    /// Initialize the module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> CausalSelfAttention<B> {
        let head_dim = self.head_dim();
        CausalSelfAttention {
            c_q: LinearConfig::new(self.n_embed, self.n_head * head_dim)
                .with_bias(false)
                .init(device),
            c_k: LinearConfig::new(self.n_embed, self.n_kv_head * head_dim)
                .with_bias(false)
                .init(device),
            c_v: LinearConfig::new(self.n_embed, self.n_kv_head * head_dim)
                .with_bias(false)
                .init(device),
            c_proj: LinearConfig::new(self.n_embed, self.n_embed)
                .with_bias(false)
                .init(device),
        }
    }
}

/// Causal Self-Attention Module
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub c_q: Linear<B>,
    pub c_k: Linear<B>,
    pub c_v: Linear<B>,
    pub c_proj: Linear<B>,
}

impl<B: Backend> CausalSelfAttentionMeta for CausalSelfAttention<B> {
    fn n_embed(&self) -> usize {
        self.c_q.weight.dims()[0]
    }

    fn n_head(&self) -> usize {
        self.c_q.weight.dims()[1] / self.head_dim()
    }

    fn n_kv_head(&self) -> usize {
        self.c_k.weight.dims()[1] / self.head_dim()
    }
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Forward Pass.
    #[allow(unused)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos_sin: (f32, f32),
    ) -> Tensor<B, 3> {
        let [b, t] = unpack_shape_contract!(
            ["batch", "time", "embed"],
            &x.dims(),
            &["batch", "time"],
            &[("embed", self.n_embed())]
        );

        let q = self
            .c_q
            .forward(x.clone())
            .reshape([b, t, self.n_head(), self.head_dim()]);
        let q = rotary::apply_rotary_embedding(q, cos_sin);
        let q = norm::rms_norm(q);
        let q = q.swap_dims(1, 2);

        let k = self
            .c_k
            .forward(x.clone())
            .reshape([b, t, self.n_kv_head(), self.head_dim()]);
        let k = rotary::apply_rotary_embedding(k, cos_sin);
        let k = norm::rms_norm(k);
        let k = k.swap_dims(1, 2);

        let v = self
            .c_v
            .forward(x)
            .reshape([b, t, self.n_kv_head(), self.head_dim()]);
        let v = v.swap_dims(1, 2);

        let y = attention::scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            None,
            ScaledDotProductAttentionConfig::new().with_enable_gqa(true),
        );

        let y = y.swap_dims(1, 2);
        let y = y.reshape([b as i32, t as i32, -1]);

        let y = self.c_proj.forward(y);

        assert_shape_contract_periodically!(
            ["batch", "time", "embed"],
            &y.dims(),
            &[("batch", b), ("time", t), ("embed", self.n_embed())]
        );

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csa_config() {
        let cfg = CausalSelfAttentionConfig::new(3, 2, 4);
        assert_eq!(cfg.n_embed, 4);
        assert_eq!(cfg.n_head, 3);
        assert_eq!(cfg.n_kv_head, 2);
        assert_eq!(cfg.n_embed(), 4);
        assert_eq!(cfg.n_head(), 3);
        assert_eq!(cfg.n_kv_head(), 2);
        assert_eq!(cfg.head_dim(), 1);
    }
}
