use std::{
    fmt::Debug,
    ops::Deref,
};

use burn::{
    module::{
        Param,
        ParamId,
        Parameter,
    },
    prelude::Backend,
    tensor::{
        BasicOps,
        Tensor,
    },
};

use crate::burn_ext::burn_desc::{
    ParamKindBinding,
    TensorDesc,
};

#[derive(Debug, Clone)]
pub struct ParamDesc<T>
where
    T: Debug + Clone + Send,
{
    param_id: ParamId,
    data: T,
}

impl<T> Deref for ParamDesc<T>
where
    T: Debug + Clone + Send,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> AsRef<T> for ParamDesc<T>
where
    T: Debug + Clone + Send,
{
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<T> ParamDesc<T>
where
    T: Debug + Clone + Send,
{
    pub fn new(
        param_id: ParamId,
        param: T,
    ) -> Self {
        Self {
            param_id,
            data: param,
        }
    }

    /// The burn `ParamId`.
    pub fn param_id(&self) -> ParamId {
        self.param_id
    }
}

impl<B, const R: usize, K> From<&Param<Tensor<B, R, K>>> for ParamDesc<TensorDesc>
where
    B: Backend,
    K: BasicOps<B>,
    burn::Tensor<B, R, K>: Parameter,
    K: ParamKindBinding,
{
    fn from(param: &Param<Tensor<B, R, K>>) -> Self {
        Self::new(param.id, TensorDesc::from(&param.val()))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        nn::LinearConfig,
        prelude::Shape,
        tensor::DType,
    };

    use super::*;
    use crate::burn_ext::burn_desc::TensorKindDesc;

    #[test]
    fn test_from_param() {
        type B = Wgpu;
        let device = Default::default();

        let linear = LinearConfig::new(2, 3).init::<B>(&device);

        let param = linear.weight;

        let param_desc: ParamDesc<TensorDesc> = (&param).into();

        assert_eq!(param_desc.param_id(), param.id);
        assert_eq!(param_desc.kind(), TensorKindDesc::Float);
        assert_eq!(param_desc.dtype(), DType::F32);
        assert_eq!(param_desc.shape(), &Shape::new([2, 3]));

        assert_eq!(param_desc.rank(), 2);
        assert_eq!(param_desc.size_estimate(), DType::F32.size() * 2 * 3);
    }
}
