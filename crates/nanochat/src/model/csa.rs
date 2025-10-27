//! # Causal Self-Attention

use crate::burn_ext::nn::embedding::rotary::RotaryEmbedding;
use crate::burn_ext::nn::functional::attention;
use crate::burn_ext::nn::functional::attention::ScaledDotProductAttentionConfig;
use crate::burn_ext::norm;
use crate::model::kvcache::KVCache;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::{Backend, Bool, Int, s};

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
    /// Validate the config.
    pub fn validate(&self) {
        assert!(
            self.n_head > 0,
            "n_head must be > 0; got n_head={}",
            self.n_head
        );
        assert!(
            self.n_embed.is_multiple_of(self.n_head),
            "n_embed ({}) must be multiple of n_head ({})",
            self.n_embed,
            self.n_head
        );
        assert!(
            self.n_embed.is_multiple_of(self.n_kv_head),
            "n_embed ({}) must be multiple of n_kv_head ({})",
            self.n_embed,
            self.n_kv_head
        );
        assert!(
            self.head_dim() > 0,
            "head_dim must be > 0; got n_embed={}/n_head={}",
            self.n_embed,
            self.n_head
        );
        assert!(
            self.n_kv_head < self.n_head,
            "n_kv_head must be < n_head; got n_kv_head={}, n_head={}",
            self.n_kv_head,
            self.n_head
        );
        assert!(
            self.n_head.is_multiple_of(self.n_kv_head),
            "n_head must be divisible by n_kv_head; got n_head={}, n_kv_head={}",
            self.n_head,
            self.n_kv_head
        );
    }
}

impl CausalSelfAttentionConfig {
    /// Initialize the module.
    pub fn init<B: Backend>(
        self,
        layer_index: usize,
        device: &B::Device,
    ) -> CausalSelfAttention<B> {
        self.validate();
        let head_dim = self.head_dim();

        CausalSelfAttention {
            layer_index,
            head_dim,
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
    pub layer_index: usize,
    pub head_dim: usize,
    pub c_q: Linear<B>,
    pub c_k: Linear<B>,
    pub c_v: Linear<B>,
    pub c_proj: Linear<B>,
}

impl<B: Backend> CausalSelfAttentionMeta for CausalSelfAttention<B> {
    fn head_dim(&self) -> usize {
        self.head_dim
    }

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
    ///
    /// # Arguments
    /// - `input`: a ``[B, T, D]`` sequence.
    /// - `re`: a rotary embedding with len ``T``.
    /// - `kv_cache`: optional KV cache.
    ///
    /// # Returns
    /// - ``[B, T, D]`` attention.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        re: &RotaryEmbedding<B>,
        kv_cache: &mut Option<&mut KVCache<B>>,
    ) -> Tensor<B, 3> {
        let [b, t_q] = unpack_shape_contract!(
            ["B", "T", "D"],
            &input.dims(),
            &["B", "T"],
            &[("D", self.n_embed())]
        );

        let q = self
            .c_q
            .forward(input.clone())
            .reshape([b, t_q, self.n_head(), self.head_dim()]);
        let q = re.apply(q);
        let q = norm::rms_norm(q);
        let q = q.swap_dims(1, 2);

        let k =
            self.c_k
                .forward(input.clone())
                .reshape([b, t_q, self.n_kv_head(), self.head_dim()]);
        let k = re.apply(k);
        let k = norm::rms_norm(k);
        let mut k = k.swap_dims(1, 2);

        let mut v = self
            .c_v
            .forward(input)
            .reshape([b, t_q, self.n_kv_head(), self.head_dim()])
            .swap_dims(1, 2);

        // B, H_?, T, D

        if let Some(kvc) = kv_cache {
            (k, v) = (*kvc).insert_kv(self.layer_index, k.clone(), v.clone());
        }
        let t_kv = k.dims()[2];

        // Number of queries in this forward pass.
        let _t_q = q.dims()[2];
        // Number of keys/values in total (in the cache + current forward pass)
        let _t_kv = k.dims()[2];

        let (attn_mask, cfg) = if kv_cache.is_none() || t_q == t_kv {
            (None, ScaledDotProductAttentionConfig::new()
                .with_is_causal(true))
        } else if t_q == 1 {
            (None, ScaledDotProductAttentionConfig::new()
                .with_is_causal(false))
        } else {
            let device = q.device();

            let mut attn_mask = Tensor::<B, 2, Bool>::empty([t_q, t_kv], &device);
            let prefix_len = t_kv - t_q;
            if prefix_len > 0 {
                attn_mask = attn_mask.slice_fill(s![.., ..prefix_len], true);
            }
            let fill = Tensor::<B, 2, Int>::ones([t_q, t_q], &device)
                .tril(-1)
                .bool();
            attn_mask = attn_mask.slice_assign(s![.., prefix_len..], fill);

            (Some(attn_mask), ScaledDotProductAttentionConfig::new()
                .with_is_causal(false))
        };

        let y =
            attention::scaled_dot_product_attention(
                q,
                k,
                v,
                None,
                attn_mask,
                cfg
                    .with_enable_gqa(self.gqa_enabled()),
            )
            .swap_dims(1, 2)
            .reshape([b as i32, t_q as i32, self.n_embed() as i32]);

        let y = self.c_proj.forward(y);

        assert_shape_contract_periodically!(
            ["B", "T", "D"],
            &y.dims(),
            &[("B", b), ("T", t_q), ("D", self.n_embed())]
        );

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn_ext::nn::embedding::rotary::RotaryEmbeddingConfig;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::Cuda;
    use burn::tensor::Distribution;

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

    #[test]
    #[allow(unused)]
    fn test_csa_forward() {
        type B = Cuda;
        let device = Default::default();

        let batch = 1;
        let seq_len = 10;

        let n_embed = 1024;
        let n_head = 128;
        let n_kv_head = 64;
        let layer_index = 12;

        let cfg = CausalSelfAttentionConfig::new(n_head, n_kv_head, n_embed);
        let csa: CausalSelfAttention<B> = cfg.init(layer_index, &device);

        let head_dim = csa.head_dim();

        let re_cfg = RotaryEmbeddingConfig::new(seq_len, csa.head_dim());
        let re: RotaryEmbedding<B> = re_cfg.init(&device);

        let input: Tensor<B, 3> =
            Tensor::random([batch, seq_len, n_embed], Distribution::Default, &device);

        let mut kv_cache: Option<&mut KVCache<B>> = None;

        let output = csa.forward(input.clone(), &re, &mut kv_cache);
        assert_shape_contract!(
            ["B", "T", "D"],
            &output.dims(),
            &[("B", batch), ("T", seq_len), ("D", n_embed)]
        );
    }
}
