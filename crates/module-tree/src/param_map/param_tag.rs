use burn::{
    Tensor,
    module::{
        Param,
        ParamId,
        Parameter,
    },
    prelude::{
        Backend,
        Shape,
    },
    tensor::{
        BasicOps,
        DType,
        Element,
        TensorKind,
    },
};

use crate::ParamKind;

/// A reference to a parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamTag {
    /// The id of the parameter.
    id: ParamId,

    /// The kind of the parameter.
    kind: ParamKind,

    /// The data type of the parameter.
    dtype: DType,

    /// The shape of the parameter.
    shape: Shape,
}

impl ParamTag {
    /// Creates a new `ParamRef`.
    pub fn new(
        id: ParamId,
        kind: ParamKind,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self {
            id,
            kind,
            dtype,
            shape,
        }
    }

    pub fn from_param<B, const D: usize, K>(
        param: &Param<Tensor<B, D, K>>,
        kind: ParamKind,
    ) -> Self
    where
        B: Backend,
        Tensor<B, D, K>: Parameter,
        K: TensorKind<B> + BasicOps<B>,
    {
        ParamTag::new(
            param.id,
            kind,
            <B as Backend>::FloatElem::dtype(),
            param.shape().clone(),
        )
    }

    /// Returns the id of the parameter.
    pub fn id(&self) -> ParamId {
        self.id
    }

    /// Returns the kind of the parameter.
    pub fn kind(&self) -> ParamKind {
        self.kind
    }

    /// Returns the data type of the parameter.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the shape of the parameter.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
