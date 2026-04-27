#![allow(unused)]

use std::{
    collections::{
        HashMap,
        HashSet,
    },
    fmt::{
        Debug,
        Display,
    },
    sync::Arc,
};

use burn::{
    module::{
        Module,
        ModuleVisitor,
        ParamId,
    },
    prelude::Backend,
    tensor::DType,
};
use xee_xpath::{
    Documents,
    Item,
    Itemable,
    Queries,
    Query,
    error::Result as SpannedResult,
    query::Convert,
};
use xot::{
    NameId,
    Node,
    Xot,
};

use crate::{
    burn_ext::burn_desc::{
        ParamDesc,
        TensorDesc,
    },
    error::{
        BunsenError,
        BunsenResult,
    },
    implementation::ModuleTreeBuilder,
    pretty_print_node,
    xee_util,
    xee_util::adapt_xee_error,
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

pub struct NodeSet<'a> {
    tree: &'a mut ModuleTree,
    nodes: Vec<Node>,
}

impl<'a> NodeSet<'a> {
    pub fn new(
        tree: &'a mut ModuleTree,
        nodes: Vec<Node>,
    ) -> Self {
        Self { tree, nodes }
    }
}

pub struct QueryBuilder<'a> {
    tree: &'a mut ModuleTree,
    expr: String,
}

const PARAM_ID_ATTR: &str = "param_id";
const SHAPE_ATTR: &str = "shape";
const RANK_ATTR: &str = "rank";
const KIND_ATTR: &str = "kind";
const DTYPE_ATTR: &str = "dtype";

impl<'a> QueryBuilder<'a> {
    pub fn new(tree: &'a mut ModuleTree) -> Self {
        Self {
            tree,
            expr: "/ModuleTree/Nodes".to_string(),
        }
    }

    pub fn expr(&self) -> String {
        self.expr.clone()
    }

    pub fn select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> QueryBuilder<'a> {
        Self {
            tree: self.tree,
            expr: format!("{}/{}", self.expr, expr.as_ref()),
        }
    }

    pub fn filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> QueryBuilder<'a> {
        Self {
            tree: self.tree,
            expr: format!("{}[{}]", self.expr, pred.as_ref()),
        }
    }

    pub fn params(self) -> Self {
        self.select("descendant-or-self::Param")
    }

    pub fn nodes(self) -> BunsenResult<NodeSet<'a>> {
        let nodes = self
            .tree
            .query_many(self.tree.root, &self.expr, |_, item| Ok(item.to_node()?))
            .map_err(|e| xee_util::adapt_xee_error(e, Some(&self.expr)))?;

        Ok(NodeSet::new(self.tree, nodes))
    }

    pub fn many<V, F>(
        &mut self,
        f: F,
    ) -> BunsenResult<Vec<V>>
    where
        F: Convert<V>,
    {
        let expr = &self.expr;
        let root = self.tree.root;

        Queries::default()
            .many(expr, f)
            .map_err(|e| adapt_xee_error(e, Some(expr)))?
            .execute(&mut self.tree.docs, root)
            .map_err(|e| adapt_xee_error(e, Some(expr)))
    }

    pub fn strings(mut self) -> BunsenResult<Vec<String>> {
        self.many(|docs, item| Ok(item.string_value(docs.xot())?))
    }

    pub fn tensor_params(mut self) -> BunsenResult<Vec<ParamDesc<TensorDesc>>> {
        let param_id_nid = self.tree.add_name(PARAM_ID_ATTR);
        let dtype_nid = self.tree.add_name(DTYPE_ATTR);
        let rank_nid = self.tree.add_name(RANK_ATTR);
        let kind_nid = self.tree.add_name(KIND_ATTR);
        let shape_nid = self.tree.add_name(SHAPE_ATTR);

        let nodes: Vec<Node> = self.many(|docs, item| Ok(item.to_node()?))?;

        let xot = self.tree.xot();
        nodes
            .into_iter()
            .map(|node| {
                let attrs = xot.attributes(node);

                let param_id: ParamId = ParamId::deserialize(
                    attrs.get(param_id_nid).expect("Param ID attribute missing"),
                );
                let tensor_desc = TensorDesc::from_strings(
                    attrs.get(kind_nid).unwrap(),
                    attrs.get(dtype_nid).unwrap(),
                    attrs.get(shape_nid).unwrap(),
                )?;

                Ok(ParamDesc::new(param_id, tensor_desc))
            })
            .collect()
    }

    pub fn map_strings<V, F>(
        mut self,
        name: &str,
        mut f: F,
    ) -> BunsenResult<Vec<V>>
    where
        F: FnMut(&str) -> BunsenResult<V>,
    {
        self.strings()?
            .into_iter()
            .map(|s| f(&s))
            .collect::<BunsenResult<Vec<V>>>()
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

    pub fn add_name(
        &mut self,
        name: &str,
    ) -> NameId {
        let xot = self.xot_mut();
        xot.add_name(name)
    }

    pub fn root(&self) -> Node {
        self.root
    }

    pub fn docs(&self) -> &Documents {
        &self.docs
    }

    pub fn docs_mut(&mut self) -> &mut Documents {
        &mut self.docs
    }

    pub fn xot(&self) -> &Xot {
        self.docs.xot()
    }

    pub fn xot_mut(&mut self) -> &mut Xot {
        self.docs.xot_mut()
    }

    pub fn query<'a>(&'a mut self) -> QueryBuilder<'a> {
        QueryBuilder::new(self)
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

        let q = Queries::default().one(expr, convert)?;

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
        let q = Queries::default().many(expr, convert)?;
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
    use std::ops::{
        Deref,
        DerefMut,
    };

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
    use crate::{
        XotDocumentsHandle,
        pretty_print_node,
        xee_util::{
            adapt_xee_error,
            pretty_errorvalue,
        },
    };

    fn print_node_query(
        mtree: &mut ModuleTree,
        selector: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let q = Queries::default().many(selector, |_docs, item| Ok(item.to_node()?))?;

        let nodes: Vec<xot::Node> = q.execute(&mut mtree.docs, mtree.root)?;

        println!("Query: {selector}");
        for (idx, node) in nodes.into_iter().enumerate() {
            print!("[{idx}]: ");
            pretty_print_node(mtree.docs.xot(), node)?;
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
    fn test_gpt() -> BunsenResult<()> {
        type B = Wgpu;
        let device = Default::default();

        let module: GPT<B> = GPTConfig::new().with_n_layer(1).init(&device);

        let mut mtree = ModuleTree::build(&module);

        println!("{:#?}", mtree);

        /*
        let ps = mtree
            .select("GPT/Vec[@name='h']//Param")
            .where_expr("@rank = 2")
            .param_ids()?;

        println!("Params: {ps:?}");
         */

        use xee_xpath::Item;

        let ids: Vec<ParamId> = mtree
            .query()
            .select("GPT/*[@name='h']")
            .params()
            .filter("@rank=2")
            .map_strings("@param_id", |s| {
                // TODO:
                // ParamId::deserialize() panics on error.
                // ParamId::try_deserialize() should exist, PR open:
                // https://github.com/tracel-ai/burn/pull/4881
                let param_id = ParamId::deserialize(&s);
                Ok(param_id)
            })?;

        println!("IDs: {ids:?}");

        let descs: Vec<ParamDesc<TensorDesc>> = mtree
            .query()
            .select("GPT/*[@name='h']")
            .params()
            .filter("@rank=2")
            .tensor_params()?;

        println!("Descs: {descs:#?}");

        let ids = descs.iter().map(|d| d.param_id()).collect::<Vec<_>>();

        println!("IDs: {ids:?}");

        Ok(())
    }
}
