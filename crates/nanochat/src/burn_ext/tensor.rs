//! # Tensor Extensions

use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::indexing::canonicalize_dim;
use burn::tensor::{AsIndex, BasicOps, Numeric};

/// Repeat Interleave.
///
/// Repeats elements of a tensor, interleaved from their existing locations.
///
/// # Arguments
/// - `input` - the input tensor.
/// - `repeats` - the number of repeats.
/// - `dim` - the dim to repeat; supports negative indexing.
///
/// # Returns
/// - the interleaved tensor.
///
/// # Examples
/// ```rust, ignore
/// type B = Wgpu;
/// let device = Default::default();
///
/// let input = Tensor::<B, 2>::from_data(
///     [
///         [0., 1., 2.],
///         [3., 4., 5.],
///     ],
///     &device);
///
/// let result: Tensor<B, 2> = repeat_interleave::<_, 2, 3, _>(input, 3, 1);
///
/// result.to_data().assert_eq(
///     &Tensor::<B, 2>::from_data(
///         [
///             [0., 0., 0., 1., 1., 1., 2., 2., 2.],
///             [3., 3., 3., 4., 4., 4., 5., 5., 5.],
///         ],
///         &device)
///     .to_data(),
///     true);
/// ```
pub fn repeat_interleave<B: Backend, const R: usize, const R2: usize, D: AsIndex>(
    input: Tensor<B, R>,
    repeats: usize,
    dim: D,
) -> Tensor<B, R> {
    let dim = canonicalize_dim(dim, R, false);

    let x: Tensor<B, R2> = input.unsqueeze_dim(dim + 1);

    let mut dims = x.dims();
    dims[dim + 1] = repeats;

    let x = x.expand(dims);

    x.flatten(dim, dim + 1)
}

/// Outer Product of two tensors.
///
/// # Arguments
/// - `a`: a ``[M]`` tensor.
/// - `b`: a ``[N]`` tensor.
///
/// # Returns
/// - a ``[M, N]`` tensor.
pub fn outer<B: Backend, K: BasicOps<B> + Numeric<B>>(
    a: Tensor<B, 1, K>,
    b: Tensor<B, 1, K>,
) -> Tensor<B, 2, K> {
    a.unsqueeze_dim::<2>(1) * b.unsqueeze_dim::<2>(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    fn test_outer() {
        type B = Wgpu;
        let device = Default::default();

        let a: Tensor<B, 1> = Tensor::arange(1..5, &device).float();
        let b: Tensor<B, 1> = Tensor::arange(1..4, &device).float();

        let res = outer(a, b);

        res.to_data().assert_eq(
            &Tensor::<B, 2>::from_data(
                [[1., 2., 3.], [2., 4., 6.], [3., 6., 9.], [4., 8., 12.]],
                &device,
            )
            .to_data(),
            true,
        )
    }

    #[test]
    fn test_repeat_interleave() {
        type B = Wgpu;
        let device = Default::default();

        let input = Tensor::<B, 2>::from_data([[0., 1., 2.], [3., 4., 5.]], &device);

        repeat_interleave::<_, 2, 3, _>(input.clone(), 3, 1)
            .to_data()
            .assert_eq(
                &Tensor::<B, 2>::from_data(
                    [
                        [0., 0., 0., 1., 1., 1., 2., 2., 2.],
                        [3., 3., 3., 4., 4., 4., 5., 5., 5.],
                    ],
                    &device,
                )
                .to_data(),
                true,
            );

        repeat_interleave::<_, 2, 3, _>(input.clone(), 3, 0)
            .to_data()
            .assert_eq(
                &Tensor::<B, 2>::from_data(
                    [
                        [0., 1., 2.],
                        [0., 1., 2.],
                        [0., 1., 2.],
                        [3., 4., 5.],
                        [3., 4., 5.],
                        [3., 4., 5.],
                    ],
                    &device,
                )
                .to_data(),
                true,
            );
    }
}
