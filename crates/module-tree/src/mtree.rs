#![allow(unused)]

use std::{
    collections::HashSet,
    fmt::{
        Debug,
        Display,
    },
};

use burn::{
    module::{
        Module,
        ModuleVisitor,
        ParamId,
    },
    prelude::Backend,
};
use xee_xpath::{
    Documents,
    Queries,
    Query,
    error::Result as SpannedResult,
    query::Convert,
};
use xot::{
    Node,
    Xot,
};

use crate::{
    error::{
        BunsenError,
        BunsenResult,
    },
    implementation::ModuleTreeBuilder,
    pretty_print_node,
    xee_util,
};

pub struct ModuleTree {
    docs: Documents,
    root: Node,
}

impl Default for ModuleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Selector builder for [`ModuleTree`].
pub struct TreeSelector<'a> {
    tree: &'a mut ModuleTree,
    expr: String,
}

impl<'a> TreeSelector<'a> {
    pub fn new(
        tree: &'a mut ModuleTree,
        expr: String,
    ) -> Self {
        Self { tree, expr }
    }

    /// The bound xpath expression.
    pub fn expr(&self) -> &str {
        &self.expr
    }

    pub fn select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> TreeSelector<'a> {
        let expr = format!("{}/{}", self.expr, expr.as_ref());
        TreeSelector::new(self.tree, expr)
    }

    pub fn where_expr<S: AsRef<str>>(
        self,
        expr: S,
    ) -> TreeSelector<'a> {
        let expr = format!("{}[{}]", self.expr, expr.as_ref());
        TreeSelector::new(self.tree, expr)
    }

    pub fn params(self) -> TreeSelector<'a> {
        self.select("descendant-or-self::Param")
    }

    fn param_nodes(&mut self) -> BunsenResult<Vec<Node>> {
        let expr = format!("/ModuleTree/Nodes/{}/descendant-or-self::Param", self.expr);

        self.tree
            .query_many(self.tree.root, &expr, |_, item| Ok(item.to_node()?))
            .map_err(|e| xee_util::adapt_xee_error(e, Some(&expr)))
    }

    pub fn param_ids(&mut self) -> BunsenResult<HashSet<ParamId>> {
        let nodes = self.param_nodes()?;

        let param_id_nid = self.tree.xot_mut().add_name("param_id");

        let mut results: HashSet<ParamId> = HashSet::with_capacity(nodes.len());
        let xot = self.tree.xot();
        for node in nodes {
            let param_id: &str = match xot.get_attribute(node, param_id_nid) {
                Some(val) => val,
                None => {
                    return Err(BunsenError::Invalid(
                        "Malformed XML: Param missing param_id attribute".into(),
                    ));
                }
            };

            // TODO: this panics on error; which is dumb.
            let param_id = ParamId::deserialize(param_id);
            results.insert(param_id);
        }

        Ok(results)
    }
}

impl ModuleTree {
    pub fn build<B: Backend, M: Module<B>>(module: &M) -> Self {
        let mut builder = ModuleTreeBuilder::default();
        module.visit(&mut builder);
        builder.build()
    }

    pub fn new() -> Self {
        let mut docs = Documents::new();
        let xot = docs.xot_mut();
        let mtree_nid = xot.add_name("ModuleTree");
        let root = xot.new_element(mtree_nid);
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

    pub fn select<'a>(
        &'a mut self,
        expr: &str,
    ) -> TreeSelector<'a> {
        TreeSelector::new(self, expr.to_string())
    }

    pub fn param_ids(&mut self) -> BunsenResult<HashSet<ParamId>> {
        self.select(".").param_ids()
    }

    fn query_one<V, F>(
        &mut self,
        root: Node,
        expr: &str,
        convert: F,
    ) -> Result<V, Box<dyn std::error::Error>>
    where
        F: Convert<V>,
    {
        let xot = self.xot_mut();
        let queries = Queries::default();

        let q = queries.one(expr, convert)?;

        q.execute(&mut self.docs, root)
            .map_err(|e| e.to_string().into())
    }

    fn query_many<V, F>(
        &mut self,
        root: Node,
        expr: &str,
        convert: F,
    ) -> SpannedResult<Vec<V>>
    where
        F: Convert<V>,
    {
        let xot = self.xot_mut();
        let queries = Queries::default();
        let q = queries.many(expr, convert)?;
        q.execute(&mut self.docs, root)
    }
}

impl Debug for ModuleTree {
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
    use zsl_chat::gpt::gpt_model::{
        GPT,
        GPTConfig,
    };

    use super::*;
    use crate::pretty_print_node;
    fn print_node_query(
        xtree: &mut ModuleTree,
        selector: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let queries = Queries::default();
        let q = queries.many(selector, |_docs, item| Ok(item.to_node()?))?;

        let nodes: Vec<xot::Node> = q.execute(&mut xtree.docs, xtree.root)?;

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

        let mut mtree = ModuleTree::build(&module);

        println!("{:#?}", mtree);

        assert_eq!(
            mtree
                .select("TestModule/*[@name='seq']/Linear/Param[@name='weight']")
                .param_ids()?,
            vec![module.seq[0].weight.id,].into_iter().collect(),
        );

        assert_eq!(
            mtree
                .select("TestModule/*[@name='seq']/Linear/Param")
                .where_expr("@name = 'weight'")
                .param_ids()?,
            vec![module.seq[0].weight.id,].into_iter().collect(),
        );

        print_node_query(&mut mtree, "//*[@id='n:2']")?;

        print_node_query(
            &mut mtree,
            "//TestModule/*[@name='tup']/Linear/Param[@rank=2]",
        )?;

        print_node_query(&mut mtree, "//TestModule/*[@name='tup']/*[2]")?;

        Ok(())
    }

    #[test]
    fn test_gpt() -> Result<(), Box<dyn std::error::Error>> {
        type B = Wgpu;
        let device = Default::default();

        let module: GPT<B> = GPTConfig::new().with_n_layer(1).init(&device);

        let mut mtree = ModuleTree::build(&module);

        println!("{:#?}", mtree);

        let muon_params = mtree
            .select("GPT/Vec[@name='h']//Param")
            .where_expr("@rank = 2")
            .param_ids()?;

        println!("Muon params: {muon_params:?}");

        Ok(())
    }
}
