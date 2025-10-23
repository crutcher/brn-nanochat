//! # Rotary Embedding

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::prelude::{Backend, s};

/// Apply rotary embedding.
///
/// # Arguments
/// - `input`: a ``[B, T, H, D]`` tensor.
/// - `cos_sin`: a tuple of ``(cos, sin)`` values.
///
/// # Returns
/// - a ``[B, T, H, D]`` tensor.
pub fn apply_rotary_embedding<B: Backend>(
    input: Tensor<B, 4>,
    cos_sin: (f32, f32),
) -> Tensor<B, 4> {
    let [b, t, h, d] = unpack_shape_contract!(["B", "T", "H", "D"], &input.dims(),);

    let dtype = input.dtype();
    let (cos, sin) = cos_sin;

    let pivot = d / 2;

    let x1 = input.clone().slice_dim(3, s![..pivot]);
    let x2 = input.clone().slice_dim(3, s![pivot..]);

    let y1 = x1.clone() * cos + x2.clone() * sin;
    let y2 = x1 * (-sin) + x2 * cos;

    let x = Tensor::cat(vec![y1, y2], 3).cast(dtype);

    assert_shape_contract_periodically!(
        ["B", "T", "H", "D"],
        &x.dims(),
        &[("B", b), ("T", t), ("H", h), ("D", d)]
    );
    x
}
