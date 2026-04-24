use burn::{
    Tensor,
    prelude::{
        Backend,
        Shape,
    },
    tensor,
    tensor::{
        BasicOps,
        DType,
    },
};

/// Encodes a description af [`burn::tensor::TensorKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TensorKindDesc {
    /// A Bool Tensor
    /// Equivalent to [`burn::tensor::Bool`].
    Bool,

    /// A Float Tensor
    /// Equivalent to [`burn::tensor::Float`].
    Float,

    /// An Int Tensor
    /// Equivalent to [`burn::tensor::Int`].
    Int,
}

/// A trait that binds a burn Tensor Kind to a `ParamKind`.
pub trait ParamKindBinding {
    const KIND: TensorKindDesc;
}

impl ParamKindBinding for tensor::Bool {
    const KIND: TensorKindDesc = TensorKindDesc::Bool;
}

impl ParamKindBinding for tensor::Float {
    const KIND: TensorKindDesc = TensorKindDesc::Float;
}

impl ParamKindBinding for tensor::Int {
    const KIND: TensorKindDesc = TensorKindDesc::Int;
}

impl TensorKindDesc {
    pub const fn for_kind<K: ParamKindBinding>() -> Self {
        K::KIND
    }
}

/// Description af a Tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDesc {
    kind: TensorKindDesc,
    dtype: DType,
    shape: Shape,
}

impl<B, const R: usize, K> From<&Tensor<B, R, K>> for TensorDesc
where
    B: Backend,
    K: BasicOps<B>,
    K: ParamKindBinding,
{
    fn from(param: &Tensor<B, R, K>) -> Self {
        Self {
            kind: TensorKindDesc::for_kind::<K>(),
            dtype: param.dtype(),
            shape: param.shape(),
        }
    }
}

impl TensorDesc {
    /// Create a new `TensorDesc`.
    pub fn new(
        kind: TensorKindDesc,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self { kind, dtype, shape }
    }

    /// The [`TensorKindDesc`] kind wrapper.
    pub fn kind(&self) -> TensorKindDesc {
        self.kind
    }

    /// The dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The rank of the shape.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// The estimated size of the tensor.
    pub fn size_estimate(&self) -> usize {
        self.dtype.size() * self.shape.num_elements()
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        tensor,
    };

    use super::*;

    #[test]
    fn test_param_kind() {
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Bool>(),
            TensorKindDesc::Bool
        );
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Float>(),
            TensorKindDesc::Float
        );
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Int>(),
            TensorKindDesc::Int
        );
    }

    #[test]
    fn test_tensor_desc() {
        type B = Wgpu;
        let device = Default::default();

        {
            // Float
            let tensor: Tensor<B, 2> = Tensor::ones([2, 3], &device);

            let desc = TensorDesc::from(&tensor);

            assert_eq!(desc.kind, TensorKindDesc::Float);
            assert_eq!(desc.dtype, DType::F32);
            assert_eq!(desc.shape, Shape::new([2, 3]));

            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), DType::F32.size() * 2 * 3);
        }

        {
            // Bool
            let tensor: Tensor<B, 2, tensor::Bool> = Tensor::zeros([2, 3], &device);

            let desc = TensorDesc::from(&tensor);

            assert_eq!(desc.kind, TensorKindDesc::Bool);
            assert_eq!(desc.dtype, tensor.dtype());
            assert_eq!(desc.shape, Shape::new([2, 3]));

            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), tensor.dtype().size() * 2 * 3);
        }

        {
            // Int
            let tensor: Tensor<B, 2, tensor::Int> = Tensor::zeros([2, 3], &device);

            let desc = TensorDesc::from(&tensor);

            assert_eq!(desc.kind, TensorKindDesc::Int);
            assert_eq!(desc.dtype, tensor.dtype());
            assert_eq!(desc.shape, Shape::new([2, 3]));

            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), tensor.dtype().size() * 2 * 3);
        }
    }
}
