use burn::{
    module::{
        Param,
        ParamId,
        Parameter,
    },
    prelude::Backend,
    tensor::{
        BasicOps,
        DType,
        Shape,
        Tensor,
    },
};

use crate::{
    ParamKindBinding,
    TensorKindDesc,
};

/// Description af a burn Param.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorParamDesc {
    param_id: ParamId,
    kind: TensorKindDesc,
    dtype: DType,
    shape: Shape,
}

impl<B, const R: usize, K> From<&Param<Tensor<B, R, K>>> for TensorParamDesc
where
    B: Backend,
    K: BasicOps<B>,
    burn::Tensor<B, R, K>: Parameter,
    K: ParamKindBinding,
{
    fn from(param: &Param<Tensor<B, R, K>>) -> Self {
        Self {
            param_id: param.id,
            kind: TensorKindDesc::for_kind::<K>(),
            dtype: param.dtype(),
            shape: param.shape(),
        }
    }
}

impl TensorParamDesc {
    /// Create a new `ParamDesc`.
    pub fn new(
        param_id: ParamId,
        kind: TensorKindDesc,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self {
            param_id,
            kind,
            dtype,
            shape,
        }
    }

    /// The burn `ParamId`.
    pub fn param_id(&self) -> ParamId {
        self.param_id
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
        nn::LinearConfig,
    };

    use super::*;

    #[test]
    fn test_from_param() {
        type B = Wgpu;
        let device = Default::default();

        let linear = LinearConfig::new(2, 3).init::<B>(&device);

        let param = linear.weight;

        let desc = TensorParamDesc::from(&param);

        assert_eq!(
            desc,
            TensorParamDesc::new(
                param.id,
                TensorKindDesc::Float,
                param.dtype(),
                param.shape()
            )
        );
    }

    #[test]
    fn test_param_desc() {
        let desc = TensorParamDesc::new(
            0.into(),
            TensorKindDesc::Float,
            DType::F32,
            Shape::new([2, 3]),
        );

        assert_eq!(desc.param_id(), 0.into());
        assert_eq!(desc.kind(), TensorKindDesc::Float);
        assert_eq!(desc.dtype(), DType::F32);
        assert_eq!(desc.shape(), &Shape::new([2, 3]));

        assert_eq!(desc.rank(), 2);
        assert_eq!(desc.size_estimate(), DType::F32.size() * 2 * 3);
    }
}
