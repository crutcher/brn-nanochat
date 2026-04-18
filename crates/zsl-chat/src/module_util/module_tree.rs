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
        /// The name of the container.
        name: String,

        /// The kind of the container.
        kind: String,
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
    fn arena(&self) -> &Arena<MTreeNodeData> {
        &self.arena
    }

    fn arena_mut(&mut self) -> &mut Arena<MTreeNodeData> {
        &mut self.arena
    }

    fn root_id(&self) -> NodeId {
        self.root_id
    }

    pub fn root(&self) -> &Node<MTreeNodeData> {
        self.arena.get(self.root_id).unwrap()
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
        let current = mtree.root_id();
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
        self.current.append_value(node, self.mtree.arena_mut())
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
        self.current = self.current.parent(self.mtree.arena()).unwrap();
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

        let root = tree.root();
        walk(0, root, tree.arena());
    }

    fn walk(
        depth: usize,
        node: &Node<MTreeNodeData>,
        arena: &Arena<MTreeNodeData>,
    ) {
        let pad = " ".repeat(depth * 2);
        println!("{}- {:?}", pad, node.get());

        let mut cur = node.first_child();
        while let Some(child_id) = cur {
            let child_node = arena.get(child_id).expect("child node not found");
            walk(depth + 1, child_node, arena);
            cur = child_node.next_sibling();
        }
    }
}
