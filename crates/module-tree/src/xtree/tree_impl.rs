#![allow(unused)]

use std::fmt::{
    Debug,
    Display,
};

use burn::{
    module::{
        Module,
        ModuleVisitor,
    },
    prelude::Backend,
};
use xee_xpath::Documents;
use xot::{
    Node,
    Xot,
};

use crate::xtree::builder::ModuleShadowTreeBuilder;

pub struct ModuleShadowTree {
    docs: Documents,
    root: Node,
}

impl Default for ModuleShadowTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleShadowTree {
    pub fn build<B: Backend, M: Module<B>>(module: M) -> Self {
        let mut builder = ModuleShadowTreeBuilder::default();
        module.visit(&mut builder);
        builder.build()
    }

    pub fn new() -> Self {
        let mut docs = Documents::new();
        let xot = docs.xot_mut();
        let mtree_name = xot.add_name("mtree");
        let root = xot.new_element(mtree_name);
        let doc = xot.new_document_with_element(root).unwrap();

        Self { docs, root }
    }

    pub fn root(&self) -> Node {
        self.root
    }

    pub fn xot(&self) -> &Xot {
        self.docs.xot()
    }

    pub fn xot_mut(&mut self) -> &mut Xot {
        self.docs.xot_mut()
    }
}

impl Debug for ModuleShadowTree {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // TODO: serialize_xml_write() directly?
        // requires fmt::Write to io::Write adaptor

        let contents = self
            .xot()
            .serialize_xml_string(
                xot::output::xml::Parameters {
                    indentation: if f.alternate() {
                        Some(Default::default())
                    } else {
                        None
                    },
                    ..Default::default()
                },
                self.root,
            )
            .map_err(|e| std::fmt::Error)?;

        f.write_str(&contents)
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
        Queries,
        Query,
    };

    use super::*;
    use crate::xtree::{
        builder::ModuleShadowTreeBuilder,
        pretty_print_node,
    };

    fn print_node_query(
        xtree: &mut ModuleShadowTree,
        selector: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let queries = Queries::default();
        let q = queries.many(selector, |_docs, item| Ok(item.to_node()))?;

        let nodes: Vec<xot::Node> = q
            .execute(&mut xtree.docs, xtree.root)?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        println!("Query: {selector}");
        for (idx, node) in nodes.into_iter().enumerate() {
            print!("[{idx}]: ");
            pretty_print_node(xtree.docs.xot(), node)?;
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
        let mut mtree = ModuleShadowTree::build(module);

        println!("{:#?}", mtree);

        print_node_query(&mut mtree, "//*[@id='1:3:1']")?;

        print_node_query(
            &mut mtree,
            "//TestModule/*[@name='tup']/Linear/Param[@rank=2]",
        )?;

        print_node_query(&mut mtree, "//TestModule/*[@name='tup']/*[2]")?;

        Ok(())
    }
}
