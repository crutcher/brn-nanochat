use burn::{
    Tensor,
    module::{
        ModuleVisitor,
        Param,
    },
    prelude::{
        Backend,
        Bool,
        Int,
    },
};

use crate::{
    ParamKind,
    param_map::{
        ParamDesc,
        ParamPath,
        ParamPathNode,
        ParamTag,
        param_map_impl::ParamMap,
    },
};

#[derive(Debug, Clone, Default)]
pub(crate) struct ParamMapBuildingVisitor<B: Backend> {
    stack: Vec<ParamPathNode>,
    param_map: ParamMap,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> ParamMapBuildingVisitor<B> {
    fn add_stack_param(
        &mut self,
        tag: ParamTag,
    ) {
        let path = ParamPath::new(self.stack.clone());
        let desc = ParamDesc { path, tag };
        self.param_map.add_param(desc);
    }

    pub(crate) fn param_map(self) -> ParamMap {
        self.param_map
    }
}

impl<B: Backend> ModuleVisitor<B> for ParamMapBuildingVisitor<B> {
    fn enter_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        self.stack.push(ParamPathNode::new(name, container_type));
    }

    fn exit_module(
        &mut self,
        _name: &str,
        _container_type: &str,
    ) {
        self.stack.pop();
    }

    fn visit_bool<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Bool>>,
    ) {
        let tag = ParamTag::from_param(param, ParamKind::Bool);
        self.add_stack_param(tag);
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D>>,
    ) {
        let tag = ParamTag::from_param(param, ParamKind::Float);
        self.add_stack_param(tag);
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Int>>,
    ) {
        let tag = ParamTag::from_param(param, ParamKind::Int);
        self.add_stack_param(tag);
    }
}
