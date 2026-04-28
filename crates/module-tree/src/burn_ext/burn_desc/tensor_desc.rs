use std::str::FromStr;

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

use crate::{
    burn_enc::shape_from_xml_attr,
    error::{
        BunsenError,
        BunsenResult,
    },
};

/// Encodes a description af [`burn::tensor::TensorKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, strum::EnumString)]
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

fn parse_dtype(dtype: &str) -> BunsenResult<DType> {
    Ok(match dtype {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "Flex32" => DType::Flex32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U64" => DType::U64,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        "Bool" => DType::Bool,
        _ => return Err(BunsenError::External(format!("Invalid dtype: {}", dtype))),
    })
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

    pub fn from_strings(
        kind: &str,
        dtype: &str,
        shape: &str,
    ) -> BunsenResult<Self> {
        let shape: Shape = shape_from_xml_attr(shape)?;

        let kind = TensorKindDesc::from_str(kind)
            .map_err(|e| BunsenError::External(format!("Invalid kind: {}", e)))?;

        let dtype = parse_dtype(dtype)?;

        Ok(Self::new(kind, dtype, shape))
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
    use burn::tensor;

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
    #[cfg(feature = "cuda")]
    fn test_tensor_desc() {
        type B = burn::backend::Cuda;
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
