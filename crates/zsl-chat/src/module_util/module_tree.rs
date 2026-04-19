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
/// Has nodes showing the structure and parameters of a model.
#[derive(Debug, Clone)]
pub struct MTree {
    arena: Arena<MTreeNode>,
    root_id: NodeId,
}

impl Default for MTree {
    fn default() -> Self {
        let mut arena = Arena::new();
        let root = arena.new_node(MTreeNode {
            name: None,
            data: MTreeNodeData::Container(ContainerData {
                kind: String::new(),
            }),
        });
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

/// A node stored in the arena. Combines the relationship name (edge label)
/// with the node's content, since indextree does not support edge data.
#[derive(Debug, Clone)]
pub struct MTreeNode {
    /// Name within the parent (None for the root).
    pub name: Option<String>,
    pub data: MTreeNodeData,
}

/// Represents a reference to a node in the module tree.
pub struct MTreeNodeRef<'a> {
    tree: &'a MTree,
    node_id: NodeId,
    node: &'a Node<MTreeNode>,
}

impl<'a> Display for MTreeNodeRef<'a> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "\"{name}\": ")?;
        }
        let n = self.node.get();
        match &n.data {
            MTreeNodeData::Container(ContainerData { kind }) => {
                let kind = if kind.starts_with("Struct:") {
                    &kind[7..]
                } else {
                    kind.as_str()
                };
                write!(f, "{}", kind)
            }
            MTreeNodeData::Param(ParamData {
                param_id,
                kind,
                dtype,
                shape,
            }) => {
                write!(f, "id={param_id} {kind:?}::{dtype:?} {:?}", &shape.dims)
            }
        }
    }
}

impl MTreeNodeRef<'_> {
    /// Get the name of this node within its parent (None for the root).
    pub fn name(&self) -> Option<&str> {
        self.node.get().name.as_deref()
    }

    /// Get the data associated with this tree node.
    pub fn data(&self) -> &MTreeNodeData {
        &self.node.get().data
    }

    /// Is this node a branch node (can it have children)?
    pub fn is_branch(&self) -> bool {
        matches!(self.data(), MTreeNodeData::Container(_))
    }

    /// Is this node a leaf node?
    pub fn is_leaf(&self) -> bool {
        matches!(self.data(), MTreeNodeData::Param(_))
    }

    /// Get the NodeIds of the children of this node.
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
    /// A container node. The root is a Container with name = None.
    Container(ContainerData),

    /// A parameter (leaf node).
    Param(ParamData),
}

impl MTreeNodeData {
    pub fn is_branch(&self) -> bool {
        matches!(self, MTreeNodeData::Container(_))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, MTreeNodeData::Param(_))
    }
}

/// The type of a container node.
///
/// Does not include a name: the name is a property of the parent/child
/// relationship and is stored in `MTreeNode`.
#[derive(Debug, Clone)]
pub struct ContainerData {
    /// The type of this container (e.g. "Vec", "Struct:Linear").
    ///
    /// Initially empty; filled in when the first child calls `enter_module`,
    /// which reports the current container's type via its `container_type`
    /// argument.
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct ParamData {
    pub param_id: ParamId,
    pub kind: ParamKind,
    pub dtype: DType,
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
        node: MTreeNode,
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
        // `container_type` is the type of the CURRENT node (the node we are
        // visiting, whose child `name` we are about to enter). Update it now
        // that we know what it is.
        if let Some(node) = self.mtree.arena.get_mut(self.current) {
            if let MTreeNodeData::Container(ref mut data) = node.get_mut().data {
                data.kind = container_type.to_string();
            }
        }

        // Create a child container. Its own kind is unknown until its children
        // call enter_module and reveal this node's type.
        self.current = self.add_child(MTreeNode {
            name: Some(name.to_string()),
            data: MTreeNodeData::Container(ContainerData {
                kind: String::new(),
            }),
        });
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
        let node = self.mtree.arena.get_mut(self.current).unwrap();
        node.get_mut().data = MTreeNodeData::Param(ParamData {
            param_id: param.id,
            kind: ParamKind::Bool,
            dtype: param.dtype(),
            shape: param.shape(),
        });
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D>>,
    ) {
        let node = self.mtree.arena.get_mut(self.current).unwrap();
        node.get_mut().data = MTreeNodeData::Param(ParamData {
            param_id: param.id,
            kind: ParamKind::Float,
            dtype: param.dtype(),
            shape: param.shape(),
        });
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Int>>,
    ) {
        let node = self.mtree.arena.get_mut(self.current).unwrap();
        node.get_mut().data = MTreeNodeData::Param(ParamData {
            param_id: param.id,
            kind: ParamKind::Int,
            dtype: param.dtype(),
            shape: param.shape(),
        });
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

    use super::*;

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        seq: Vec<Linear<B>>,
    }

    impl<B: Backend> TestModule<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                seq: vec![
                    LinearConfig::new(10, 10).init(device),
                    LinearConfig::new(10, 23).init(device),
                ],
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
