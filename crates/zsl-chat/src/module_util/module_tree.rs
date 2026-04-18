#![allow(unused)]

use std::{
    fmt::Display,
    marker::PhantomData,
};

use burn::{
    Tensor,
    module::{
        Module,
        ModuleVisitor,
        Param,
        ParamId,
    },
    prelude::{
        Backend,
        Bool,
        Int,
        Shape,
    },
    tensor::DType,
};
use indextree::{
    Arena,
    Node,
    NodeId,
};

use crate::module_util::kinds::ParamKind;

/// A shadow tree of a module.
///
/// Has nodes shodiwng the structure and parameters of a model.
#[derive(Debug, Clone)]
pub struct MTree {
    arena: Arena<MTreeNodeData>,
    root_id: NodeId,
}

impl Default for MTree {
    fn default() -> Self {
        let mut arena = Arena::new();
        let root = arena.new_node(MTreeNodeData::Root);
        Self {
            arena,
            root_id: root,
        }
    }
}

impl MTree {
    /// Builds a module tree from a module.
    pub fn build<B: Backend, M: Module<B>>(module: &M) -> Self {
        let mut visitor = MTreeBuildingVistior::<B>::default();
        module.visit(&mut visitor);
        visitor.build()
    }

    /// Returns the root node of the tree.
    pub fn root(&self) -> MTreeNodeRef<'_> {
        MTreeNodeRef {
            tree: self,
            node_id: self.root_id,
            node: self.arena.get(self.root_id).unwrap(),
        }
    }
}

/// Represents a reference to a node in the module tree.
pub struct MTreeNodeRef<'a> {
    tree: &'a MTree,
    node_id: NodeId,
    node: &'a Node<MTreeNodeData>,
}

impl<'a> Display for MTreeNodeRef<'a> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self.data() {
            MTreeNodeData::Root => write!(f, "/"),
            MTreeNodeData::Container(ContainerData { kind, name }) => {
                let kind = if kind.starts_with("Struct:") {
                    &kind[7..]
                } else {
                    kind
                };
                write!(f, "{name:?}: {kind}")
            }
            MTreeNodeData::Param(ParamData {
                param_id,
                kind,
                dtype,
                shape,
            }) => {
                write!(
                    f,
                    "{}: {:?} ({:?}, {:?})",
                    param_id, kind, dtype, &shape.dims
                )
            }
        }
    }
}

impl MTreeNodeRef<'_> {
    /// Get the data associated with this tree node.
    pub fn data(&self) -> &MTreeNodeData {
        self.node.get()
    }

    /// Is this node a branch node?
    /// (Can it have children?)
    pub fn is_branch(&self) -> bool {
        self.data().is_branch()
    }

    /// Is this node a leaf node?
    pub fn is_leaf(&self) -> bool {
        self.data().is_leaf()
    }

    /// Get the NodeIds of the children of this node.
    ///
    /// Will return an empty vector if this node has no children; or is a leaf.
    pub fn child_ids(&self) -> Vec<NodeId> {
        let mut node_ids = vec![];
        let mut cur = self.node.first_child();
        while let Some(id) = cur {
            let node = self.tree.arena.get(id).unwrap();
            node_ids.push(id);
            cur = node.next_sibling();
        }
        node_ids
    }

    /// Get an iterator over the children of this node.
    pub fn children(&self) -> impl Iterator<Item = MTreeNodeRef<'_>> {
        self.child_ids().into_iter().map(move |id| MTreeNodeRef {
            tree: self.tree,
            node_id: id,
            node: self.tree.arena.get(id).unwrap(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MTreeNodeData {
    /// The module root.
    Root,

    /// A named container.
    Container(ContainerData),

    /// A parameter.
    Param(ParamData),
}

impl MTreeNodeData {
    /// Whether this node/node type can have children.
    pub fn is_branch(&self) -> bool {
        !self.is_leaf()
    }

    /// Whether this node/node type is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, MTreeNodeData::Param { .. })
    }
}

#[derive(Debug, Clone)]
pub struct ContainerData {
    /// The name of the container.
    pub name: String,

    /// The kind of the container.
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct ParamData {
    /// The id of the parameter.
    pub param_id: ParamId,

    /// The kind of the parameter.
    pub kind: ParamKind,

    /// The data type of the parameter.
    pub dtype: DType,

    /// The shape of the parameter.
    pub shape: Shape,
}
/// A visitor that builds a module tree.
#[derive(Debug, Clone)]
struct MTreeBuildingVistior<B: Backend> {
    mtree: MTree,
    current: NodeId,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for MTreeBuildingVistior<B> {
    fn default() -> Self {
        let mtree: MTree = Default::default();
        let current = mtree.root_id;
        Self {
            mtree,
            current,
            phantom: PhantomData::<B>,
        }
    }
}

impl<B: Backend> MTreeBuildingVistior<B> {
    fn add_child(
        &mut self,
        node: MTreeNodeData,
    ) -> NodeId {
        self.current.append_value(node, &mut self.mtree.arena)
    }

    pub fn build(self) -> MTree {
        self.mtree
    }
}

impl<B: Backend> ModuleVisitor<B> for MTreeBuildingVistior<B> {
    fn enter_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        self.current = self.add_child(MTreeNodeData::Container(ContainerData {
            name: name.to_string(),
            kind: container_type.to_string(),
        }));
    }

    fn exit_module(
        &mut self,
        _name: &str,
        _container_type: &str,
    ) {
        self.current = self.current.parent(&self.mtree.arena).unwrap();
    }

    fn visit_bool<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Bool>>,
    ) {
        self.add_child(MTreeNodeData::Param(ParamData {
            param_id: param.id,
            kind: ParamKind::Bool,
            dtype: param.dtype(),
            shape: param.shape(),
        }));
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D>>,
    ) {
        self.add_child(MTreeNodeData::Param(ParamData {
            param_id: param.id,
            kind: ParamKind::Float,
            dtype: param.dtype(),
            shape: param.shape(),
        }));
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Int>>,
    ) {
        self.add_child(MTreeNodeData::Param(ParamData {
            param_id: param.id,
            kind: ParamKind::Int,
            dtype: param.dtype(),
            shape: param.shape(),
        }));
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
    };
    use indextree::{
        Arena,
        Node,
        macros::tree,
    };

    use super::*;

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
    fn test_demo() {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let mtree = MTree::build(&module);

        walk(0, &mtree.root());
    }

    fn walk<'a>(
        depth: usize,
        node: &MTreeNodeRef<'a>,
    ) {
        let pad = " ".repeat(depth * 2);
        println!("{}- {}", pad, node);

        for child in node.children() {
            walk(depth + 1, &child);
        }
    }
}
