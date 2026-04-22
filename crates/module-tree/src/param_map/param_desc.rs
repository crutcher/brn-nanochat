use burn::{
    module::ParamId,
    prelude::Shape,
    tensor::DType,
};

use crate::{
    ParamKind,
    param_map::{
        ParamPath,
        ParamTag,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamDesc {
    pub path: ParamPath,
    pub tag: ParamTag,
}

impl ParamDesc {
    pub fn path(&self) -> &ParamPath {
        &self.path
    }

    pub fn tag(&self) -> &ParamTag {
        &self.tag
    }

    pub fn kind(&self) -> ParamKind {
        self.tag.kind()
    }

    pub fn id(&self) -> ParamId {
        self.tag.id()
    }

    pub fn dtype(&self) -> DType {
        self.tag.dtype()
    }

    pub fn shape(&self) -> &Shape {
        self.tag.shape()
    }
}
