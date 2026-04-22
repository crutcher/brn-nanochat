#![allow(unused)]

use std::fmt::format;

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
        Float,
        Int,
    },
    tensor::{
        DType,
        Shape,
    },
};
use xot::{
    Node,
    Xot,
};

use crate::ParamKind;

pub struct XTreeBuilder<B: Backend> {
    xot: Xot,
    doc: Node,
    root: Node,

    node_names: Vec<String>,
    node_types: Vec<String>,
    stack: Vec<Node>,

    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for XTreeBuilder<B> {
    fn default() -> Self {
        let mut xot = Xot::new();
        let mtree_name = xot.add_name("mtree");
        let root = xot.new_element(mtree_name);
        let doc = xot.new_document_with_element(root).unwrap();
        Self {
            xot,
            doc,
            root,
            node_names: Default::default(),
            node_types: Default::default(),
            stack: vec![root],
            phantom: Default::default(),
        }
    }
}

impl<B: Backend> XTreeBuilder<B> {
    fn add_param(
        &mut self,
        param_id: ParamId,
        param_kind: ParamKind,
        dtype: DType,
        shape: Shape,
    ) {
        let parent = *self.stack.last().unwrap();

        let elem_id = self.xot.add_name("Param");

        let node = self.xot.new_element(elem_id);

        let format_id = param_id.to_string();
        let id_id = self.xot.add_name("id");
        self.xot.set_attribute(node, id_id, format_id);

        let format_kind = format!("{:?}", param_kind);
        let kind_id = self.xot.add_name("kind");
        self.xot.set_attribute(node, kind_id, format_kind);

        let format_dtype = format!("{:?}", dtype);
        let dtype_id = self.xot.add_name("dtype");
        self.xot.set_attribute(node, dtype_id, format_dtype);

        let format_shape = shape.to_string();
        let shape_id = self.xot.add_name("shape");
        self.xot.set_attribute(node, shape_id, format_shape);

        if let Some(name) = self.node_names.last() {
            let name_id = self.xot.add_name("name");
            self.xot.set_attribute(node, name_id, name);
        }

        self.xot.append(parent, node);
    }
}

impl<B: Backend> ModuleVisitor<B> for XTreeBuilder<B> {
    fn enter_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        let parent = *self.stack.last().unwrap();

        self.node_names.push(name.to_string());
        self.node_types.push(container_type.to_string());

        if self.node_types.len() + 1 > self.stack.len() {
            let type_name = self.node_types.last().unwrap();
            let type_id = self.xot.add_name(type_name);

            let node = self.xot.new_element(type_id);
            self.xot.append(parent, node);

            if self.node_names.len() > 1 {
                let name_attr = self.xot.add_name("name");
                let node_name = &self.node_names[self.node_types.len() - 2];

                self.xot.set_attribute(node, name_attr, node_name);
            }

            self.stack.push(node);
        }
    }

    fn exit_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        self.node_names.pop();
        self.node_types.pop();
        self.stack.pop();
    }

    fn visit_bool<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Bool>>,
    ) {
        self.add_param(param.id, ParamKind::Bool, param.dtype(), param.shape());
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Float>>,
    ) {
        self.add_param(param.id, ParamKind::Float, param.dtype(), param.shape());
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Int>>,
    ) {
        self.add_param(param.id, ParamKind::Int, param.dtype(), param.shape());
    }
}

#[cfg(test)]
#[allow(unused)]
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
    use xee_xpath::{
        Documents,
        Queries,
        Query,
    };
    use xot::{
        Node,
        Xot,
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

    fn pretty_print(
        xot: &Xot,
        node: Node,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut output_params = xot::output::xml::Parameters {
            indentation: Some(Default::default()),
            ..Default::default()
        };
        println!("{}", xot.serialize_xml_string(output_params, node)?);

        Ok(())
    }

    #[test]
    fn test_builder() -> Result<(), Box<dyn std::error::Error>> {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let mut builder = XTreeBuilder::default();
        module.visit(&mut builder);

        pretty_print(&builder.xot, builder.doc)?;
        Ok(())
    }

    #[test]
    fn test_xot_construction() -> Result<(), Box<dyn std::error::Error>> {
        let mut xot = Xot::new();

        let root = xot.parse("<p>Example</p>")?;
        let doc_el = xot.document_element(root)?;
        let txt = xot.first_child(doc_el).unwrap();
        let txt_value = xot.text_mut(txt).unwrap();
        txt_value.set("Hello, world!");

        assert_eq!(xot.to_string(root)?, "<p>Hello, world!</p>");

        pretty_print(&xot, root)?;

        Ok(())
    }

    #[test]
    fn test_xee_xpath() -> Result<(), Box<dyn std::error::Error>> {
        // create a new documents object
        let mut documents = Documents::new();
        // load a document from a string
        let doc = documents
            .add_string(
                "http://example.com".try_into().unwrap(),
                "<root><bar>foo</bar></root>",
            )
            .unwrap();

        // create a new queries object
        let queries = Queries::default();

        // create a query expecting a single value in the result sequence
        // try to convert this value into a Rust `String`
        let q = queries.one("/root/string()", |_, item| {
            Ok(item.try_into_value::<String>()?)
        })?;

        // when we execute the query, we need to pass a mutable reference to the
        // documents, and the item against which we want to query. We can also
        // pass in a document handle, as we do here
        let r = q.execute(&mut documents, doc)?;
        assert_eq!(r, "foo");

        pretty_print(documents.xot(), documents.document_node(doc).unwrap())?;

        Ok(())
    }
}
