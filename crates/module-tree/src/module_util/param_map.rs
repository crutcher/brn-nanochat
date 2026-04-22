//! Parameter Map
use std::collections::BTreeMap;

use burn::{
    Tensor,
    module::{
        Module,
        ModuleVisitor,
        Param,
        ParamId,
        Parameter,
    },
    prelude::{
        Backend,
        Bool,
        Int,
        Shape,
    },
    tensor::{
        BasicOps,
        DType,
        Element,
        TensorKind,
    },
};

use crate::module_util::kinds::ParamKind;

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
}

/// Represents a node in a module tree path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamPathNode {
    /// The name of the node.
    name: String,

    /// The name of the container type of the node.
    container: String,
}

impl ParamPathNode {
    /// Creates a new `ModulePathNode`.
    pub fn new(
        name: &str,
        container: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            container: container.to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn container(&self) -> &str {
        &self.container
    }
}

/// Represents a path in a module tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamPath(Vec<ParamPathNode>);

impl ParamPath {
    pub fn new(nodes: Vec<ParamPathNode>) -> Self {
        assert!(!nodes.is_empty());
        Self(nodes)
    }

    pub fn nodes(&self) -> &[ParamPathNode] {
        &self.0
    }

    pub fn push(
        &mut self,
        node: ParamPathNode,
    ) {
        self.0.push(node);
    }

    pub fn path_str(&self) -> String {
        self.0
            .iter()
            .map(|n| n.name.clone())
            .collect::<Vec<_>>()
            .join(".")
    }
}

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
        self.tag.kind
    }

    pub fn id(&self) -> ParamId {
        self.tag.id
    }

    pub fn dtype(&self) -> DType {
        self.tag.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.tag.shape
    }
}

/*
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PathCursor {
    pub path: Arc<ParamPath>,
    pub idx: usize,
}

impl PathCursor {
    pub fn new(path: Arc<ParamPath>) -> Self {
        Self { path, idx: 0 }
    }

    pub fn is_leaf(&self) -> bool {
        self.idx == self.path.nodes().len() - 1
    }

    pub fn current(&self) -> Option<&ParamPathNode> {
        self.path.nodes().get(self.idx)
    }

    pub fn advance(
        &self,
        step: usize,
    ) -> Option<Self> {
        if self.idx + step >= self.path.nodes().len() {
            None
        } else {
            Some(Self {
                path: self.path.clone(),
                idx: self.idx + step,
            })
        }
    }
}
 */

/// A map from module paths to parameter kinds.
#[derive(Debug, Clone, Default)]
pub struct ParamMap {
    params: BTreeMap<ParamPath, ParamDesc>,
}

impl ParamMap {
    /// Collects the parameter map from a module.
    pub fn collect<M: Module<B>, B: Backend>(module: &M) -> Self {
        let mut visitor = ParamMapBuildingVisitor::<B>::default();
        module.visit(&mut visitor);
        visitor.param_map
    }

    /// Adds a parameter to the map.
    pub fn add_param(
        &mut self,
        desc: ParamDesc,
    ) {
        self.params.insert(desc.path.clone(), desc);
    }

    /// Returns an iterator over the parameter map.
    pub fn iter(&self) -> impl Iterator<Item = (&ParamPath, &ParamDesc)> {
        self.params.iter()
    }

    /// Returns the number of parameters in the map.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
struct ParamMapBuildingVisitor<B: Backend> {
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

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        nn::{
            Linear,
            LinearConfig,
        },
        prelude::Backend,
    };

    use super::*;

    #[test]
    fn test_param_kind() {
        assert_eq!(ParamKind::Bool, ParamKind::Bool);
        assert_ne!(ParamKind::Bool, ParamKind::Float);
    }

    #[test]
    fn test_param_ref() {
        let ref1 = ParamTag::new(1.into(), ParamKind::Bool, DType::Bool, [2, 3].into());
        let ref1_dup = ParamTag::new(1.into(), ParamKind::Bool, DType::Bool, [2, 3].into());
        let ref1_cp = ref1.clone();

        assert_eq!(ref1, ref1_dup);
        assert_eq!(ref1, ref1_cp);

        assert_eq!(ref1.id(), 1.into());
        assert_eq!(ref1.kind(), ParamKind::Bool);

        let ref2 = ParamTag::new(2.into(), ParamKind::Float, DType::F32, [2, 3].into());
        let ref3 = ParamTag::new(3.into(), ParamKind::Int, DType::I32, [2, 3].into());

        assert_eq!(ref2.id(), 2.into());
        assert_eq!(ref2.kind(), ParamKind::Float);

        assert_eq!(ref3.id(), 3.into());
        assert_eq!(ref3.kind(), ParamKind::Int);

        assert_ne!(ref1, ref2);
        assert_ne!(ref1, ref3);
    }

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        seq: Vec<Linear<B>>,
    }

    impl<B: Backend> TestModule<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                seq: vec![LinearConfig::new(10, 10).init(device)],
            }
        }
    }

    #[test]
    fn test_module_path() {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let _param_map = ParamMap::collect(&module);

        /*
        assert_eq!(
            &param_map.iter().collect::<Vec<_>>(),
            &vec![
                (
                    &ParamPath(vec![
                        ParamPathNode::new("seq", "Struct:TestModule"),
                        ParamPathNode::new("0", "Vec"),
                        ParamPathNode::new("bias", "Struct:Linear"),
                    ]),
                    &ParamTag::new(
                        module.seq[0].bias.as_ref().unwrap().id,
                        ParamKind::Float,
                        <B as Backend>::FloatElem::dtype(),
                        [10].into()
                    )
                ),
                (
                    &ParamPath(vec![
                        ParamPathNode::new("seq", "Struct:TestModule"),
                        ParamPathNode::new("0", "Vec"),
                        ParamPathNode::new("weight", "Struct:Linear"),
                    ]),
                    &ParamTag::new(
                        module.seq[0].weight.id,
                        ParamKind::Float,
                        <B as Backend>::FloatElem::dtype(),
                        [10, 10].into()
                    )
                ),
            ]
        );

         */
    }
}
