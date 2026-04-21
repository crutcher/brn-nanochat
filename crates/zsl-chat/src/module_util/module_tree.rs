#![allow(unused)]

use std::{
    fmt::Display,
    marker::PhantomData,
    ops::Index,
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
use debug_ignore::DebugIgnore;
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
            tree: self.into(),
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

    pub fn try_container(&self) -> Option<&ContainerData> {
        if let MTreeNodeData::Container(data) = self {
            Some(data)
        } else {
            None
        }
    }

    pub fn as_param(&self) -> Option<&ParamData> {
        if let MTreeNodeData::Param(data) = self {
            Some(data)
        } else {
            None
        }
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
    kind: String,
}

impl ContainerData {
    /// Access the container's type.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Is this a struct?
    pub fn is_struct(&self) -> bool {
        self.kind.starts_with("Struct:")
    }

    /// If this is a struct, return its name.
    pub fn try_struct(&self) -> Option<&str> {
        self.kind.strip_prefix("Struct:")
    }

    /// If this is a struct, return its name.
    /// Panics if this is not a struct.
    pub fn expect_struct(&self) -> &str {
        self.try_struct().unwrap()
    }

    /// Is this a builtin?
    pub fn is_builtin(&self) -> bool {
        !self.is_struct()
    }

    /// If this is a builtin, return its name.
    pub fn try_builtin(&self) -> Option<&str> {
        if self.is_struct() {
            None
        } else {
            Some(&self.kind)
        }
    }

    /// If this is a builtin, return its name.
    /// Panics if this is not a builtin.
    pub fn expect_builtin(&self) -> &str {
        self.try_builtin().unwrap()
    }

    /// Is this a `Vec`?
    pub fn is_vec(&self) -> bool {
        self.kind == "Vec"
    }

    /// Is this a `Tuple`?
    pub fn is_tuple(&self) -> bool {
        self.kind == "Tuple"
    }

    /// Is this a `Array`?
    pub fn is_array(&self) -> bool {
        self.kind == "Array"
    }

    /// Is this a builtin sequence-type?
    /// (Does it have dense numeric index-named children?)
    pub fn is_sequence(&self) -> bool {
        self.is_vec() || self.is_tuple() || self.is_array()
    }
}

#[derive(Debug, Clone)]
pub struct ParamData {
    param_id: ParamId,
    kind: ParamKind,
    dtype: DType,
    shape: Shape,
}

impl ParamData {
    /// Access the parameter's ID.
    pub fn param_id(&self) -> ParamId {
        self.param_id
    }

    /// Access the parameter's kind.
    pub fn kind(&self) -> ParamKind {
        self.kind
    }

    /// Access the parameter's dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Access the parameter's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Represents a reference to a node in the module tree.
#[derive(Debug, Clone)]
pub struct MTreeNodeRef<'a> {
    tree: DebugIgnore<&'a MTree>,
    node_id: NodeId,
    node: &'a Node<MTreeNode>,
}

impl<'a> PartialEq for MTreeNodeRef<'a> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.node_id == other.node_id
    }
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
                let kind = if let Some(stripped) = kind.strip_prefix("Struct:") {
                    stripped
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

    /// View the container data (if this is a container node).
    pub fn try_container(&self) -> Option<&ContainerData> {
        self.data().try_container()
    }

    /// View the container data (if this is a container node).
    /// Panics if this is not a container node.
    pub fn expect_container(&self) -> &ContainerData {
        self.try_container().unwrap()
    }

    /// View the parameter data (if this is a param node).
    pub fn try_param(&self) -> Option<&ParamData> {
        self.data().as_param()
    }

    /// View the parameter data (if this is a param node).
    /// Panics if this is not a param node.
    pub fn expect_param(&self) -> &ParamData {
        self.try_param().unwrap()
    }

    /// Get the number of children of this node.
    /// Leaf nodes always have zero children.
    pub fn len(&self) -> usize {
        self.child_ids().len()
    }

    /// Is this node empty (no children)?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the parent of this node, if it has one.
    pub fn parent(&self) -> Option<MTreeNodeRef<'_>> {
        self.node.parent().map(|id| MTreeNodeRef {
            tree: self.tree,
            node_id: id,
            node: self.tree.arena.get(id).unwrap(),
        })
    }

    /// Get the `NodeIds` of the children of this node.
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

    /// Get a child node by name, if it exists.
    pub fn try_child(
        &self,
        name: &str,
    ) -> Option<MTreeNodeRef<'_>> {
        self.children().find(|child| child.name() == Some(name))
    }

    /// Get a child node by name.
    /// Panics if the child does not exist.
    pub fn expect_child(
        &self,
        name: &str,
    ) -> MTreeNodeRef<'_> {
        match self.try_child(name) {
            Some(child) => child,
            None => panic!("child {name:?} not found"),
        }
    }

    /// Get a child node by index, if it exists.
    pub fn try_index(
        &self,
        idx: usize,
    ) -> Option<MTreeNodeRef<'_>> {
        let name = idx.to_string();
        self.try_child(&name)
    }

    /// Get a child node by index.
    /// Panics if the index is out of bounds.
    pub fn expect_index(
        &self,
        idx: usize,
    ) -> MTreeNodeRef<'_> {
        let name = idx.to_string();
        self.expect_child(&name)
    }
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
        if let Some(node) = self.mtree.arena.get_mut(self.current)
            && let MTreeNodeData::Container(ref mut data) = node.get_mut().data
        {
            data.kind = container_type.to_string();
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
        tup: (Linear<B>, Linear<B>),
        arr: [Linear<B>; 1],
    }

    impl<B: Backend> TestModule<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                seq: vec![LinearConfig::new(10, 10).init(device)],
                tup: (
                    LinearConfig::new(10, 10).init(device),
                    LinearConfig::new(10, 23).init(device),
                ),
                arr: [LinearConfig::new(10, 10).init(device)],
            }
        }
    }

    #[test]
    fn test_demo() {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let mtree = MTree::build(&module);

        let root = mtree.root();

        assert_eq!(root.name(), None);
        assert_eq!(root.is_branch(), true);
        assert_eq!(root.is_leaf(), false);
        assert_eq!(
            root.try_container().unwrap().try_struct().unwrap(),
            "TestModule"
        );

        assert_eq!(root.children().count(), 3);

        let [seq, tup, arr] = root.children().collect::<Vec<_>>().try_into().unwrap();

        {
            assert_eq!(seq.name(), Some("seq"));
            let cont = seq.expect_container();
            assert_eq!(cont.expect_builtin(), "Vec");
            assert_eq!(cont.is_vec(), true);
            assert_eq!(cont.is_sequence(), true);

            assert_eq!(seq.len(), 1);
            assert_eq!(seq.is_empty(), false);

            let [linear] = seq.children().collect::<Vec<_>>().try_into().unwrap();
            assert_eq!(seq.expect_index(0), linear);
            assert_eq!(seq.expect_child("0"), linear);
            assert_eq!(linear.name(), Some("0"));

            assert_eq!(linear.expect_container().expect_struct(), "Linear");

            let w = linear.expect_child("weight");
            assert_eq!(w.name(), Some("weight"));
            assert_eq!(w.is_leaf(), true);
            {
                let wparam = w.expect_param();
                assert_eq!(wparam.param_id(), module.seq[0].weight.id);
                assert_eq!(wparam.kind(), ParamKind::Float);
                assert_eq!(wparam.dtype(), DType::F32);
                assert_eq!(wparam.shape().dims(), [10, 10]);
            }

            let b = linear.expect_child("bias");
            assert_eq!(b.name(), Some("bias"));
            assert_eq!(b.is_leaf(), true);
            {
                let bparam = b.expect_param();
                assert_eq!(
                    bparam.param_id(),
                    (&module.seq[0].bias).as_ref().unwrap().id
                );
                assert_eq!(bparam.kind(), ParamKind::Float);
                assert_eq!(bparam.dtype(), DType::F32);
                assert_eq!(bparam.shape().dims(), [10]);
            }
        }

        {
            assert_eq!(tup.name(), Some("tup"));
            let cont = tup.expect_container();
            assert_eq!(cont.expect_builtin(), "Tuple");
            assert_eq!(cont.is_tuple(), true);
            assert_eq!(cont.is_sequence(), true);
        }

        {
            assert_eq!(arr.name(), Some("arr"));
            let cont = arr.expect_container();
            assert_eq!(cont.expect_builtin(), "Array");
            assert_eq!(cont.is_array(), true);
            assert_eq!(cont.is_sequence(), true);
        }

        walk(0, &root);
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
