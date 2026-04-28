#![allow(unused)]

use std::fmt::{
    Debug,
    Display,
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
    Itemable,
    Queries,
    Query,
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
        TensorParamDesc,
    },
    constants::{
        DTYPE_ATTR,
        KIND_ATTR,
        MODULE_TREE_ELEM,
        PARAM_ELEM,
        PARAM_ID_ATTR,
        RANK_ATTR,
        SHAPE_ATTR,
        STRUCTURE_ELEM,
    },
    error::BunsenResult,
    implementation::ModuleTreeBuilder,
    pretty_print_node,
    xee_util,
    xee_util::adapt_xee_error,
};

pub const MODULE_TREE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct ModuleTree {
    docs: Documents,
    root: Node,
}

impl Debug for ModuleTree {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str("ModuleTree {\n")?;
        for line in self.to_xml().lines() {
            writeln!(f, "  {line}")?;
        }
        f.write_str("}")
    }
}

impl ModuleTree {
    pub fn build<B: Backend, M: Module<B>>(module: &M) -> Self {
        let mut builder = ModuleTreeBuilder::default();
        module.visit(&mut builder);
        builder.build()
    }

    /// Create a new/empty module tree.
    pub(crate) fn new() -> Self {
        let mut docs = Documents::new();
        let xot = docs.xot_mut();
        let mtree_nid = xot.add_name(MODULE_TREE_ELEM);
        let root = xot.new_element(mtree_nid);
        let doc = xot.new_document_with_element(root).unwrap();

        let version_nid = xot.add_name("version");
        xot.set_attribute(root, version_nid, MODULE_TREE_VERSION);

        Self { docs, root }
    }

    /// Serialize the module tree to an XML string.
    pub fn to_xml(&self) -> String {
        self.docs
            .xot()
            .serialize_xml_string(
                xot::output::xml::Parameters {
                    indentation: Some(Default::default()),
                    ..Default::default()
                },
                self.root,
            )
            .unwrap()
    }

    /// Bind (add/lookup) a local (no namespace) name in the [`xot`] arena.
    ///
    /// # Arguments
    /// * `name` - a string name.
    ///
    /// # Returns
    /// A [`NameId`]
    pub(crate) fn bind_local_name(
        &mut self,
        name: &str,
    ) -> NameId {
        let xot = self.xot_mut();
        xot.add_name(name)
    }

    /// Bind a list of local names to [`NameId`]s.
    ///
    /// See [`bind_local_name`].
    pub(crate) fn bind_local_names<const N: usize>(
        &mut self,
        names: [&str; N],
    ) -> [NameId; N] {
        names.map(|name| self.bind_local_name(name))
    }

    /// The root [`Node`] document node of the module tree.
    ///
    /// This is only useful with the XML apis.
    pub(crate) fn root(&self) -> Node {
        self.root
    }

    /// A const view of the [`xee_xpath`] [`Documents`] arena.
    pub fn docs(&self) -> &Documents {
        &self.docs
    }

    /// A mut view of the [`xee_xpath`] [`Documents`] arena.
    pub fn docs_mut(&mut self) -> &mut Documents {
        &mut self.docs
    }

    /// Internal. Shorthand access to the [`xot`] arena.
    pub(crate) fn xot(&self) -> &Xot {
        self.docs.xot()
    }

    /// Internal. Shorthand access to the mutable [`xot`] arena.
    pub(crate) fn xot_mut(&mut self) -> &mut Xot {
        self.docs.xot_mut()
    }

    /// Iterate over [`ParamId`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`ModuleTreeQuery::params`].
    ///
    /// # Returns
    /// `Ok(impl Iterator<Item = ParamId>)` on success, `Err(e)` on errors.
    ///
    /// ## Example
    /// ```rust,ignore
    /// let param_ids: HashSet<ParamId> = mtree
    ///     .param_ids()?
    ///     .collect();
    ///
    /// // Is equivalent to:
    /// let param_ids: HashSet<ParamId> = mtree
    ///     .query()
    ///     // .params() is implicit to [`ModuleTreeQuery::to_param_ids`],
    ///     // equivalent to: .select("descendant-or-self::Param")
    ///     .to_param_ids()?
    ///     .collect();
    /// ```
    pub fn param_ids(&mut self) -> BunsenResult<impl Iterator<Item = ParamId>> {
        self.query().to_param_ids()
    }

    /// Iterate over [`TensorParamDesc`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`ModuleTreeQuery::params`].
    ///
    /// # Returns
    /// `Ok(impl Iterator<Item = TensorParamDesc>)` on success, `Err(e)` on
    /// errors.
    ///
    /// ## Example
    /// ```rust,ignore
    /// let descs: Vec<ParamDesc<TensorDesc>> = mtree
    ///     .param_descs()?
    ///     .collect();
    ///
    /// // Is equivalent to:
    /// let descs: Vec<ParamDesc<TensorDesc>> = mtree
    ///     .query()
    ///     // .params() is implicit to [`ModuleTreeQuery::to_param_descs`],
    ///     // equivalent to: .select("descendant-or-self::Param")
    ///     .to_param_descs()?
    ///     .collect();
    /// ```
    pub fn param_descs(&mut self) -> BunsenResult<impl Iterator<Item = TensorParamDesc>> {
        self.query().to_param_descs()
    }

    /// Create a new default [`ModuleTreeQuery`] for this module tree.
    ///
    /// The query builder has a fluent api to incrementally refine a query.
    /// It begins with broad selection over the entire module structure.
    pub fn query<'a>(&'a mut self) -> ModuleTreeQuery<'a> {
        ModuleTreeQuery::new(self)
    }

    /// Query a sub-tree.
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear");
    /// ```
    pub fn select<'a>(
        &'a mut self,
        expr: &str,
    ) -> ModuleTreeQuery<'a> {
        ModuleTreeQuery::new(self).select(expr)
    }

    /// Query a sub-tree.
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear");
    /// ```
    pub fn try_select<'a>(
        &'a mut self,
        expr: &str,
    ) -> BunsenResult<ModuleTreeQuery<'a>> {
        ModuleTreeQuery::new(self).try_select(expr)
    }

    /// Query all parameters of a subtree.
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select_params("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear")
    ///     .params();
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descedant-or-self::Param");
    /// ```
    pub fn select_params<'a>(
        &'a mut self,
        expr: &str,
    ) -> ModuleTreeQuery<'a> {
        self.select(expr).params()
    }

    /// Query all parameters of a subtree.
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select_params("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear")
    ///     .params();
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descedant-or-self::Param");
    /// ```
    pub fn try_select_params<'a>(
        &'a mut self,
        expr: &str,
    ) -> BunsenResult<ModuleTreeQuery<'a>> {
        Ok(self.try_select(expr)?.params())
    }

    /// Return an iterator over the parameter [`ParamId`]s of a subtree.
    ///
    /// # Returns
    /// `Ok(impl Iterator<Item = ParamId>)` on success, `Err(e)` on `XPath`
    /// errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .select_param_ids("GPT/Linear")?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .select_params("GPT/Linear")
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .query()
    ///     .select("GPT/Linear")
    ///     .params()
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descendant-or-self::Param")
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descendant-or-self::Param")
    ///     .to_param_descs()?
    ///     .map(|d| d.param_id())
    ///     .collect();
    /// ```
    pub fn select_param_ids(
        &mut self,
        expr: &str,
    ) -> BunsenResult<impl Iterator<Item = ParamId>> {
        self.try_select_params(expr)?.to_param_ids()
    }
}

/// A query builder for a [`ModuleTree`].
///
/// This works in two phases:
/// 1. A fluent api to incrementally refine an `XPath` expression.
/// 2. Various execution/output runners to run that expression.
#[derive(Debug)]
pub struct ModuleTreeQuery<'a> {
    tree: &'a mut ModuleTree,
    expr: String,
}

impl<'a> ModuleTreeQuery<'a> {
    /// Create a new [`ModuleTreeQuery`].
    ///
    /// See: [`ModuleTree::query`].
    pub fn new(tree: &'a mut ModuleTree) -> Self {
        Self {
            tree,
            expr: format!("/{MODULE_TREE_ELEM}/{STRUCTURE_ELEM}"),
        }
    }

    /// Get the current `XPath` expression.
    pub fn expr(&self) -> &String {
        &self.expr
    }

    fn try_append_expr(
        mut self,
        expr: &str,
    ) -> BunsenResult<Self> {
        let expr = format!("{}{}", self.expr, expr);

        Queries::default()
            .many(&expr, |_, _| Ok(()))
            .map_err(|e| adapt_xee_error(e, Some(&expr)))?;

        Ok(Self { expr, ..self })
    }

    fn append_expr(
        mut self,
        expr: &str,
    ) -> Self {
        self.try_append_expr(expr)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Refine the current selection by appending an `XPath` path expression.
    ///
    /// If the expression was "E", the new expression will be "{E}/{expr}".
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    pub fn select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> ModuleTreeQuery<'a> {
        self.append_expr(format!("/{}", expr.as_ref()).as_str())
    }

    /// Refine the current selection by appending an `XPath` path expression.
    ///
    /// If the expression was "E", the new expression will be "{E}/{expr}".
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> BunsenResult<ModuleTreeQuery<'a>> {
        self.try_append_expr(format!("/{}", expr.as_ref()).as_str())
    }

    /// Refine the current selection by appending an `XPath` predicate
    /// expression.
    ///
    /// Predicate expressions filter the current node-set, keeping only those
    /// nodes for which each branch of the predicate expression evaluates to
    /// true.
    ///
    /// If the expression was "E", the new expression will be "{E}[{expr}]".
    ///
    /// # Example
    /// * `filter("@name='foo'")` - select only nodes with a "name" attribute
    ///   equal to "foo".
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    pub fn filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> ModuleTreeQuery<'a> {
        self.append_expr(format!("[{}]", pred.as_ref()).as_str())
    }

    /// Refine the current selection by appending an `XPath` predicate
    /// expression.
    ///
    /// Predicate expressions filter the current node-set, keeping only those
    /// nodes for which each branch of the predicate expression evaluates to
    /// true.
    ///
    /// If the expression was "E", the new expression will be "{E}[{expr}]".
    ///
    /// # Example
    /// * `filter("@name='foo'")` - select only nodes with a "name" attribute
    ///   equal to "foo".
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> BunsenResult<ModuleTreeQuery<'a>> {
        self.try_append_expr(format!("[{}]", pred.as_ref()).as_str())
    }

    /// Recursively select all parameter elements in the current context.
    ///
    /// This is the `descendent-or-self::Param` operation.
    pub fn params(self) -> Self {
        self.select(format!("descendant-or-self::{PARAM_ELEM}"))
    }

    /// Execute a [`xee_xpath::Queries::many`] on the current selection.
    pub fn execute_many<V, F>(
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

    /// Iterate over [`TensorParamDesc`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`Self::params`].
    ///
    /// # Returns
    /// `Ok(impl Iterator<Item = TensorParamDesc>)` on success, `Err(e)` on
    /// errors.
    pub fn to_param_descs(mut self) -> BunsenResult<impl Iterator<Item = TensorParamDesc>> {
        let [param_id_nid, dtype_nid, rank_nid, kind_nid, shape_nid] = self
            .tree
            .bind_local_names([PARAM_ID_ATTR, DTYPE_ATTR, RANK_ATTR, KIND_ATTR, SHAPE_ATTR]);

        let mut query = self.params();

        let nodes: Vec<Node> = query.execute_many(|docs, item| Ok(item.to_node()?))?;

        let xot = query.tree.docs.xot();

        Ok(nodes
            .into_iter()
            .map(|node| {
                let attrs = xot.attributes(node);

                // TODO: Extract, real errors.
                let param_id: ParamId =
                    ParamId::deserialize(attrs.get(param_id_nid).unwrap_or_else(|| {
                        panic!("{PARAM_ELEM}/{PARAM_ID_ATTR} attribute missing")
                    }));
                let tensor_desc = TensorDesc::from_strings(
                    attrs.get(kind_nid).unwrap(),
                    attrs.get(dtype_nid).unwrap(),
                    attrs.get(shape_nid).unwrap(),
                )?;

                Ok(ParamDesc::new(param_id, tensor_desc))
            })
            .collect::<BunsenResult<Vec<ParamDesc<TensorDesc>>>>()?
            .into_iter())
    }

    /// Iterate over [`ParamId`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`Self::params`].
    ///
    /// # Returns
    /// `Ok(impl Iterator<Item = ParamId>)` on success, `Err(e)` on errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let param_ids : HashSet<ParamId> = query
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = query
    ///     .to_param_descs()?
    ///     .map(|d| d.param_id())
    ///     .collect();
    /// ```
    pub fn to_param_ids(mut self) -> BunsenResult<impl Iterator<Item = ParamId>> {
        Ok(self.to_param_descs()?.map(|d| d.param_id()))
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use burn::nn::{
        Linear,
        LinearConfig,
    };
    use zsl_chat::gpt::gpt_model::{
        GPT,
        GPTConfig,
    };

    use super::*;
    use crate::pretty_print_node;

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
    #[cfg(feature = "cuda")]
    fn test_gpt() -> BunsenResult<()> {
        type B = burn::backend::Cuda;
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
            .select_params("GPT/*[@name='h']")
            .filter("@rank=2")
            .to_param_ids()?
            .collect();

        println!("IDs: {ids:?}");

        let descs: Vec<ParamDesc<TensorDesc>> = mtree
            .select_params("GPT/*[@name='h']")
            .filter("@rank=2")
            .to_param_descs()?
            .collect();

        println!("Descs: {descs:#?}");

        let ids = descs.iter().map(|d| d.param_id()).collect::<Vec<_>>();

        println!("IDs: {ids:?}");

        Ok(())
    }
}
