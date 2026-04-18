#![allow(unused)]

use std::marker::PhantomData;

use burn::{
    Tensor,
    module::{
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

use crate::module_util::param_map::ParamKind;

#[derive(Debug, Clone)]
pub enum MTreeNodeData {
    Root,

    Container {
        /// The kind of the container.
        kind: String,

        /// The name of the container.
        name: String,
    },

    Param {
        /// The id of the parameter.
        param_id: ParamId,

        /// The kind of the parameter.
        kind: ParamKind,

        /// The data type of the parameter.
        dtype: DType,

        /// The shape of the parameter.
        shape: Shape,
    },
}

impl MTreeNodeData {
    pub fn is_container(&self) -> bool {
        match self {
            MTreeNodeData::Root { .. } | MTreeNodeData::Container { .. } => true,
            _ => false,
        }
    }

    pub fn is_leaf(&self) -> bool {
        !self.is_container()
    }
}

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
    pub fn root(&self) -> MTreeNode<'_> {
        MTreeNode {
            tree: self,
            node_id: self.root_id,
            node: self.arena.get(self.root_id).unwrap(),
        }
    }
}

pub struct MTreeNode<'a> {
    tree: &'a MTree,
    node_id: NodeId,
    node: &'a Node<MTreeNodeData>,
}

impl MTreeNode<'_> {
    pub fn data(&self) -> &MTreeNodeData {
        self.node.get()
    }

    pub fn is_container(&self) -> bool {
        self.data().is_container()
    }

    pub fn is_leaf(&self) -> bool {
        self.data().is_leaf()
    }

    pub fn children(&self) -> impl Iterator<Item = MTreeNode<'_>> {
        let mut node_ids = vec![];

        let mut cur = self.node.first_child();
        while let Some(id) = cur {
            let node = self.tree.arena.get(id).unwrap();
            node_ids.push(id);
            cur = node.next_sibling();
        }

        node_ids.into_iter().map(move |id| MTreeNode {
            tree: self.tree,
            node_id: id,
            node: self.tree.arena.get(id).unwrap(),
        })
    }
}

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
    fn mtree(&self) -> &MTree {
        &self.mtree
    }

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
        self.current = self.add_child(MTreeNodeData::Container {
            name: name.to_string(),
            kind: container_type.to_string(),
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
        self.add_child(MTreeNodeData::Param {
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
        self.add_child(MTreeNodeData::Param {
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
        self.add_child(MTreeNodeData::Param {
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
        module::Module,
        nn::{
            Linear,
            LinearConfig,
        },
        prelude::Backend,
    };
    use indextree::{
        Arena,
        Node,
    };

    use crate::module_util::module_tree::{
        MTreeBuildingVistior,
        MTreeNode,
        MTreeNodeData,
    };

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

        let mut tree_builder: MTreeBuildingVistior<B> = MTreeBuildingVistior::default();
        module.visit(&mut tree_builder);

        let tree = tree_builder.build();

        walk(0, &tree.root());
    }

    fn walk<'a>(
        depth: usize,
        node: &MTreeNode<'a>,
    ) {
        let pad = " ".repeat(depth * 2);
        println!("{}- {:?}", pad, node.data());

        for child in node.children() {
            walk(depth + 1, &child);
        }
    }
}
