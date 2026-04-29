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
        Float,
        Int,
    },
};
use xot::{
    Node,
    Xot,
};

use crate::{
    modules::reflection::{
        XmlModuleTree,
        module_visitors::type_util,
        xml_support::names::{
            CLASS_ATTR,
            DTYPE_ATTR,
            ID_ATTR,
            KIND_ATTR,
            NAME_ATTR,
            PARAM_ELEM,
            PARAM_ID_ATTR,
            RANK_ATTR,
            SHAPE_ATTR,
            STRUCTURE_ELEM,
        },
    },
    tensors::TensorParamDesc,
    zspace::shape_to_xml_attr,
};

/// [`ModuleVisitor`] builder for a [`XmlModuleTree`].
pub struct ModuleTreeBuilder<B: Backend> {
    mtree: XmlModuleTree,

    depth: usize,
    base: Node,
    stack: Vec<Node>,
    pending_name: Option<String>,

    next_id: usize,

    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> ModuleTreeBuilder<B> {
    /// Build a [`XmlModuleTree`] from a [`Module`].
    pub fn build<M: Module<B>>(module: &M) -> XmlModuleTree {
        let mut builder = Self::new();
        module.visit(&mut builder);
        builder.mtree
    }

    fn new() -> Self {
        let mut mtree = XmlModuleTree::new();
        let root = mtree.root();

        let nodes_nid = mtree.bind_local_name(STRUCTURE_ELEM);
        let xot = mtree.xot_mut();
        let base = xot.new_element(nodes_nid);

        xot.append(root, base).unwrap();

        ModuleTreeBuilder::<B> {
            mtree,
            depth: 0,
            base,
            pending_name: None,
            stack: vec![],
            next_id: 0,
            phantom: Default::default(),
        }
    }

    fn xot(&self) -> &Xot {
        self.mtree.xot()
    }

    fn xot_mut(&mut self) -> &mut Xot {
        self.mtree.xot_mut()
    }

    fn add_param_desc(
        &mut self,
        param_desc: TensorParamDesc,
    ) {
        let node = self.new_child(*self.stack.last().unwrap(), PARAM_ELEM);
        self.set_idents(node);

        self.set_attribute(node, PARAM_ID_ATTR, param_desc.param_id().to_string());
        self.set_attribute(node, CLASS_ATTR, "tensor");

        // Should kind be <Type kind="Float" dtype="F32" />?
        self.set_attribute(node, KIND_ATTR, format!("{:?}", param_desc.kind()));
        self.set_attribute(node, DTYPE_ATTR, format!("{:?}", param_desc.dtype()));

        // Should shape be <Shape rank="1" dims="[10, 2]" />?
        self.set_attribute(node, SHAPE_ATTR, shape_to_xml_attr(param_desc.shape()));
        self.set_attribute(
            node,
            RANK_ATTR,
            param_desc.shape().clone().rank().to_string(),
        );
    }

    fn new_child<N: AsRef<str>>(
        &mut self,
        parent: Node,
        name: N,
    ) -> Node {
        let node = self.new_element(name);
        self.xot_mut().append(parent, node).unwrap();
        node
    }

    fn new_element<N: AsRef<str>>(
        &mut self,
        name: N,
    ) -> Node {
        let name_nid = self.mtree.bind_local_name(name.as_ref());
        self.xot_mut().new_element(name_nid)
    }

    fn set_idents(
        &mut self,
        node: Node,
    ) {
        self.next_id += 1;
        let id = format!("n:{:X}", self.next_id);
        self.set_attribute(node, ID_ATTR, id);

        if let Some(name) = self.pending_name.take()
            && !self.parent_is_sequence()
        {
            self.set_attribute(node, NAME_ATTR, name);
        }
    }

    fn set_attribute<N: AsRef<str>, V: AsRef<str>>(
        &mut self,
        node: Node,
        name: N,
        value: V,
    ) {
        let attr_nid = self.mtree.bind_local_name(name.as_ref());
        self.xot_mut().set_attribute(node, attr_nid, value.as_ref());
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

            let (cls, elem_name) = type_util::parse_container_type(container_type);
            let _elem_nid = self.mtree.bind_local_name(&elem_name);

            let elem_node = self.new_child(parent, elem_name);

            self.set_idents(elem_node);
            self.set_attribute(elem_node, CLASS_ATTR, cls);

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
        _name: &str,
        _container_type: &str,
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
