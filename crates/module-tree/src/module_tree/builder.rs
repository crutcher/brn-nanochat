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
    },
    prelude::{
        Backend,
        Bool,
        Int,
    },
    tensor::DType,
};
use indextree::NodeId;

use crate::{
    module_tree::mtree::{
        ContainerData,
        MTree,
        MTreeNode,
        MTreeNodeData,
        ParamData,
    },
    param_kind::ParamKind,
};

/// A visitor that builds a module tree.
#[derive(Debug, Clone)]
pub(crate) struct MTreeBuildingVistior<B: Backend> {
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
    use crate::module_tree::mtree::MTreeNodeRef;

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
        assert_eq!(root.parent(), None);

        assert_eq!(root.is_branch(), true);
        assert_eq!(root.is_leaf(), false);
        assert_eq!(root.expect_container().expect_struct(), "TestModule");

        assert_eq!(root.children().count(), 3);
        {
            let seq = root.expect_child("seq");
            assert_eq!(seq.parent(), Some(root));

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
            let tup = root.expect_child("tup");
            assert_eq!(tup.parent(), Some(root));

            let cont = tup.expect_container();
            assert_eq!(cont.expect_builtin(), "Tuple");
            assert_eq!(cont.is_tuple(), true);
            assert_eq!(cont.is_sequence(), true);
        }

        {
            let arr = root.expect_child("arr");
            assert_eq!(arr.parent(), Some(root));

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
