//! # Attention Extensions

use crate::burn_ext::tensor;
use burn::Tensor;
use burn::config::Config;
use burn::prelude::{Backend, Bool, Int};
use burn::tensor::DType;
use burn::tensor::activation::softmax;

#[derive(Config, Debug, Copy)]
pub struct ScaledDotProductAttentionConfig {
    /// Causal or not.
    #[config(default = "false")]
    pub is_causal: bool,

    /// Enable Group Query Attention.
    #[config(default = "false")]
    pub enable_gqa: bool,

    /// Scale factor; always enabled if set to `Some(_)`.
    #[config(default = "None")]
    pub scale: Option<f32>,

    /// Dropout rate; always enabled if set to `Some(_)`.
    #[config(default = "None")]
    pub dropout: Option<f32>,
}

/// Computes scaled dot product attention.
///
/// # Arguments
/// - `q`: the query tensor.
/// - `k`: the key tensor.
/// - `v`: the value tensor.
/// - `bias`: optional additive bias.
/// - `mask`: optional bias mask.
/// - `config`: attention config.
///
/// # Returns
/// - the attention result.
pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    bias: Option<Tensor<B, 2>>,
    mask: Option<Tensor<B, 2, Bool>>,
    config: ScaledDotProductAttentionConfig,
) -> Tensor<B, 4> {
    let device = q.device();
    let dtype = q.dtype();

    let l = q.dims()[2];
    let s = k.dims()[2];

    let scale_factor = config.scale.unwrap_or(1.0 / (q.dims()[3] as f32).sqrt());

    let attn_bias = scaled_dot_product_attention_bias(l, s, bias, mask, config, dtype, &device);

    let mut k = k;
    let mut v = v;

    if config.enable_gqa {
        let k_repeats = q.dims()[1] / k.dims()[1];
        k = tensor::repeat_interleave::<B, 4, 5, _>(k, k_repeats, 1);

        let v_repeats = q.dims()[1] / v.dims()[1];
        v = tensor::repeat_interleave::<B, 4, 5, _>(v, v_repeats, 1);
    }

    let attn_weight = q.matmul(k).swap_dims(2, 3) * scale_factor;
    let attn_weight = attn_weight + attn_bias.unsqueeze();
    let attn_weight = softmax(attn_weight, 3);

    attn_weight.matmul(v)
}

/// Build the Attention Bias for [`scaled_dot_product_attention`].
///
/// # Arguments
/// - `l`: the query time dimension.
/// - `s`: the key time dimension.
/// - `bias`: optional additive bias.
/// - `mask`: optional bias mask.
/// - `dtype`: the desired dtype of the output tensor.
/// - `device`: the target device of the bias.
///
/// # Returns
/// - a ``[l, s]`` attention bias tensor.
pub fn scaled_dot_product_attention_bias<B: Backend>(
    l: usize,
    s: usize,
    bias: Option<Tensor<B, 2>>,
    mask: Option<Tensor<B, 2, Bool>>,
    config: ScaledDotProductAttentionConfig,
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut attn_bias = Tensor::<B, 2>::zeros([l, s], device).cast(dtype);
    if config.is_causal {
        attn_bias = attn_bias.mask_fill(
            Tensor::<B, 2, Int>::ones([l, s], device)
                .tril(0)
                .bool()
                .bool_not(),
            f32::NEG_INFINITY,
        );
    }
    if let Some(bias) = bias {
        attn_bias = attn_bias + bias;
    }
    if let Some(mask) = mask {
        attn_bias = attn_bias.mask_fill(mask.bool_not(), f32::NEG_INFINITY);
    }
    attn_bias
}
