#![allow(unused)]
use burn::{
    Tensor,
    module::{
        ModuleVisitor,
        Param,
    },
    prelude::{
        Backend,
        Bool,
        Float,
        Int,
    },
};
use xot::{
    Node,
    Xot,
};

use crate::{
    ModuleTree,
    burn_ext::burn_desc::{
        ParamDesc,
        TensorDesc,
    },
    type_util,
    type_util::parse_container_type,
};

pub struct ModuleTreeBuilder<B: Backend> {
    xtree: ModuleTree,

    depth: usize,
    base: Node,
    stack: Vec<Node>,
    pending_name: Option<String>,

    next_id: usize,

    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for ModuleTreeBuilder<B> {
    fn default() -> Self {
        let mut xtree = ModuleTree::new();
        let root = xtree.root();

        let xot = xtree.xot_mut();
        let nodes_nid = xot.add_name("Nodes");
        let base = xot.new_element(nodes_nid);

        xot.append(root, base);

        Self {
            xtree,
            depth: 0,
            base,
            pending_name: None,
            stack: vec![],
            next_id: 0,
            phantom: Default::default(),
        }
    }
}

impl<B: Backend> ModuleTreeBuilder<B> {
    pub fn build(self) -> ModuleTree {
        self.xtree
    }

    fn xot(&self) -> &Xot {
        self.xtree.xot()
    }

    fn xot_mut(&mut self) -> &mut Xot {
        self.xtree.xot_mut()
    }

    fn debug_stack(&self) -> Vec<String> {
        let xot = self.xot();
        self.stack
            .iter()
            .map(|n| {
                let elem = xot.element(*n).unwrap();
                let name = elem.name();
                xot.local_name_str(name).to_string()
            })
            .collect()
    }

    fn add_param_desc(
        &mut self,
        param_desc: ParamDesc<TensorDesc>,
    ) {
        let elem_node = self.new_child(*self.stack.last().unwrap(), "Param");
        self.set_idents(elem_node);

        self.set_attribute(elem_node, "param_id", param_desc.param_id().to_string());

        // Should kind be <Type kind="Float" dtype="F32" />?
        self.set_attribute(elem_node, "kind", format!("{:?}", param_desc.kind()));
        self.set_attribute(elem_node, "dtype", format!("{:?}", param_desc.dtype()));

        // Should shape be <Shape rank="1" dims="[10, 2]" />?
        self.set_attribute(
            elem_node,
            "shape",
            param_desc
                .shape()
                .clone()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<String>>()
                .join(" "),
        );
        self.set_attribute(
            elem_node,
            "rank",
            param_desc.shape().clone().rank().to_string(),
        );
    }

    fn new_child<N: AsRef<str>>(
        &mut self,
        parent: Node,
        name: N,
    ) -> Node {
        let node = self.new_element(name);
        self.xot_mut().append(parent, node);
        node
    }

    fn new_element<N: AsRef<str>>(
        &mut self,
        name: N,
    ) -> Node {
        let xot = self.xot_mut();
        let name_nid = xot.add_name(name.as_ref());
        xot.new_element(name_nid)
    }

    fn set_idents(
        &mut self,
        node: Node,
    ) {
        self.next_id += 1;
        let id = format!("n:{:X}", self.next_id);
        self.set_attribute(node, "id", id);

        if let Some(name) = self.pending_name.take()
            && !self.parent_is_sequence()
        {
            self.set_attribute(node, "name", name);
        }
    }

    fn set_attribute<N: AsRef<str>, V: AsRef<str>>(
        &mut self,
        node: Node,
        name: N,
        value: V,
    ) {
        let xot = self.xot_mut();
        let id = xot.add_name(name.as_ref());
        xot.set_attribute(node, id, value.as_ref());
    }

    fn parent_is_sequence(&self) -> bool {
        let xot = self.xot();
        let parent = self.stack.last().unwrap();
        let pelem = xot.element(*parent).unwrap();
        let pname = pelem.name();
        let parent_type = xot.local_name_str(pname);
        type_util::type_is_sequence(parent_type)
    }
}

impl<B: Backend> ModuleVisitor<B> for ModuleTreeBuilder<B> {
    fn enter_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        if self.depth == self.stack.len() {
            // enter_module is called on each *child* of a Module.
            // This means that the first time we learn the type of a Module,
            // we are seeing the name of its first child.
            //
            // If the current depth is the same as the length of the stack,
            // then we need to create a new container.

            // If the stack is empty, then this is the root node;
            // and we need to attach it to the document.
            let parent = self.stack.last().copied().unwrap_or(self.base);

            let xot = self.xot_mut();

            let (cls, elem_name) = type_util::parse_container_type(container_type);
            let elem_nid = xot.add_name(&elem_name);

            let xot = self.xot_mut();
            let elem_node = xot.new_element(elem_nid);
            xot.append(parent, elem_node);

            self.set_idents(elem_node);
            self.set_attribute(elem_node, "class", cls);

            // There's no way to determine which enum case we are in.
            // if cls == "enum" {
            //   self.set_attribute(node, "case", ???);
            // }

            self.stack.push(elem_node);
        }

        self.depth += 1;
        self.pending_name = Some(name.to_string());
    }

    fn exit_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        if self.depth < self.stack.len() {
            self.pending_name = None;
            self.stack.pop();
        }
        self.depth -= 1;
    }

    fn visit_bool<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Bool>>,
    ) {
        self.add_param_desc(param.into());
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Float>>,
    ) {
        self.add_param_desc(param.into());
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Int>>,
    ) {
        self.add_param_desc(param.into());
    }
}
