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
use xee_xpath::Documents;
use xot::{
    NameId,
    Node,
    Xot,
};

use crate::ParamKind;

pub struct XTreeBuilder<B: Backend> {
    documents: Documents,
    doc: Node,
    root: Node,

    depth: usize,
    stack: Vec<Node>,
    pending_name: Option<String>,

    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for XTreeBuilder<B> {
    fn default() -> Self {
        let mut documents = Documents::new();
        let mtree_name = documents.xot_mut().add_name("mtree");
        let root = documents.xot_mut().new_element(mtree_name);
        let doc = documents.xot_mut().new_document_with_element(root).unwrap();
        Self {
            documents,
            doc,
            root,
            depth: 0,
            pending_name: None,
            stack: vec![root],
            phantom: Default::default(),
        }
    }
}

const SEQUENCE_TYPES: &[&str] = &["Vec", "Array", "Tuple"];

fn is_sequence_type(name: &str) -> bool {
    SEQUENCE_TYPES.contains(&name)
}

impl<B: Backend> XTreeBuilder<B> {
    fn debug_stack(&self) -> Vec<String> {
        let xot = self.documents.xot();
        self.stack
            .iter()
            .map(|n| {
                let elem = xot.element(*n).unwrap();
                let name = elem.name();
                xot.local_name_str(name).to_string()
            })
            .collect()
    }

    fn add_param(
        &mut self,
        param_id: ParamId,
        param_kind: ParamKind,
        dtype: DType,
        shape: Shape,
    ) {
        let node = self.new_child(*self.stack.last().unwrap(), "Param");

        self.set_attribute(node, "id", self.make_id());

        self.maybe_name(node);

        self.set_attribute(node, "param_id", param_id.to_string());

        // Should kind be <Type kind="Float" dtype="F32" />?
        self.set_attribute(node, "kind", format!("{:?}", param_kind));
        self.set_attribute(node, "dtype", format!("{:?}", dtype));

        // Should shape be <Shape rank="1" dims="[10, 2]" />?
        self.set_attribute(node, "shape", shape.to_string());
        self.set_attribute(node, "rank", shape.rank().to_string());

        self.stack.push(node);
    }

    fn new_child<N: AsRef<str>>(
        &mut self,
        parent: Node,
        name: N,
    ) -> Node {
        let node = self.new_element(name);
        self.documents.xot_mut().append(parent, node);
        node
    }

    fn new_element<N: AsRef<str>>(
        &mut self,
        name: N,
    ) -> Node {
        let id = self.documents.xot_mut().add_name(name.as_ref());
        self.documents.xot_mut().new_element(id)
    }

    fn set_attribute<N: AsRef<str>, V: AsRef<str>>(
        &mut self,
        node: Node,
        name: N,
        value: V,
    ) {
        let id = self.documents.xot_mut().add_name(name.as_ref());
        self.documents
            .xot_mut()
            .set_attribute(node, id, value.as_ref());
    }

    fn parent_is_sequence(&self) -> bool {
        let parent = self.stack.last().unwrap();
        let pelem = self.documents.xot().element(*parent).unwrap();
        let pname = pelem.name();
        let parent_type = self.documents.xot().local_name_str(pname);
        is_sequence_type(parent_type)
    }

    fn maybe_name(
        &mut self,
        node: Node,
    ) {
        if let Some(name) = self.pending_name.take() {
            if self.parent_is_sequence() {
                return;
            }

            self.set_attribute(node, "name", name);
        }
    }

    fn name_container(
        &mut self,
        container_type: &str,
    ) -> NameId {
        let elem_name = if let Some(name) = container_type.strip_prefix("Struct:") {
            name
        } else {
            container_type
        };
        self.documents.xot_mut().add_name(elem_name)
    }

    fn make_id(&self) -> String {
        let xot = self.documents.xot();
        self.stack
            .iter()
            .map(|n| xot.children(*n).count().to_string())
            .collect::<Vec<_>>()
            .join(":")
    }
}

fn is_struct(name: &str) -> bool {
    name.starts_with("Struct:")
}

impl<B: Backend> ModuleVisitor<B> for XTreeBuilder<B> {
    fn enter_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        self.depth += 1;

        if self.depth == self.stack.len() {
            let elem_type = self.name_container(container_type);
            let parent = *self.stack.last().unwrap();
            let node = self.documents.xot_mut().new_element(elem_type);
            self.documents.xot_mut().append(parent, node);

            self.maybe_name(node);

            self.set_attribute(node, "id", self.make_id());

            self.set_attribute(
                node,
                "class",
                if is_struct(container_type) {
                    "struct"
                } else {
                    "builtin"
                },
            );

            self.stack.push(node);
        }

        self.pending_name = Some(name.to_string());
    }

    fn exit_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        self.depth -= 1;
        if self.stack.len() > self.depth + 1 {
            self.pending_name = None;
            self.stack.pop();
        }
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

    fn print_node_query<B: Backend>(
        builder: &mut XTreeBuilder<B>,
        selector: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let queries = Queries::default();
        let q = queries.many(selector, |_docs, item| Ok(item.to_node()))?;

        let nodes: Vec<xot::Node> = q
            .execute(&mut builder.documents, builder.doc)?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        println!("Query: {selector}");
        for (idx, node) in nodes.into_iter().enumerate() {
            print!("[{idx}]: ");
            pretty_print(&builder.documents.xot(), node)?;
        }

        Ok(())
    }

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
    fn test_builder() -> Result<(), Box<dyn std::error::Error>> {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let mut builder = XTreeBuilder::default();
        module.visit(&mut builder);

        pretty_print(&builder.documents.xot(), builder.doc)?;

        print_node_query(&mut builder, "//*[@id='1:3:1']")?;

        print_node_query(
            &mut builder,
            "//TestModule/*[@name='tup']/Linear/Param[@rank=2]",
        )?;

        print_node_query(&mut builder, "//TestModule/*[@name='tup']/*[2]")?;

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
