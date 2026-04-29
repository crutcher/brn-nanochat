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
    error::{
        ErrorValue,
        Result as SpannedResult,
    },
    query::Convert,
};
use xot::{
    NameId,
    Node,
    Xot,
};

use crate::{
    burn_ext::{
        ParamDesc,
        TensorDesc,
        TensorParamDesc,
    },
    errors::BunsenResult,
    module_visitors::ModuleTreeBuilder,
    xml_support::{
        adapt_xee_error,
        names::{
            DTYPE_ATTR,
            KIND_ATTR,
            MODULE_TREE_ELEM,
            PARAM_ELEM,
            PARAM_ID_ATTR,
            RANK_ATTR,
            SHAPE_ATTR,
            STRUCTURE_ELEM,
        },
    },
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
        f.write_str("ModuleTree {")?;
        if f.alternate() {
            f.write_str("\n")?;
            for line in self.to_xml(true).lines() {
                writeln!(f, "  {line}")?;
            }
        } else {
            f.write_str(self.to_xml(false).as_str())?;
        }
        f.write_str("}")
    }
}

impl ModuleTree {
    /// Build a [`ModuleTree`] for a [`Module`].
    pub fn build<B: Backend, M: Module<B>>(module: &M) -> Self {
        ModuleTreeBuilder::build(module)
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
    pub fn to_xml(
        &self,
        pretty: bool,
    ) -> String {
        self.docs
            .xot()
            .serialize_xml_string(
                xot::output::xml::Parameters {
                    indentation: if pretty {
                        Some(Default::default())
                    } else {
                        None
                    },
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

    /// Iterate over string fragments for the current expression matches.
    ///
    /// # Arguments
    /// * `pretty` - pretty-print/indent the xml fragments.
    pub fn to_fragments(
        &mut self,
        pretty: bool,
    ) -> BunsenResult<impl Iterator<Item = String>> {
        use xee_xpath::Item;

        let output_params = xot::output::xml::Parameters {
            indentation: if pretty {
                Some(Default::default())
            } else {
                None
            },
            ..Default::default()
        };

        let res = self.execute_many(
            |docs: &mut Documents, item: &Item| -> SpannedResult<String> {
                let xot: &xot::Xot = docs.xot();

                match item {
                    Item::Node(node) => xot
                        .serialize_xml_string(output_params.clone(), *node)
                        .map(|s| s.trim().to_string())
                        .map_err(|e| ErrorValue::from(e).into()),
                    _ => Ok(item.string_value(xot)?),
                }
            },
        )?;

        Ok(res.into_iter())
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

    #[test]
    #[cfg(feature = "cuda")]
    fn test_debug_cuda() {
        test_debug::<burn::backend::Cuda>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_debug_wgpu() {
        test_debug::<burn::backend::Wgpu>();
    }

    fn test_debug<B: Backend>() {
        let device = Default::default();
        let module: Linear<B> = LinearConfig::new(2, 3).init(&device);

        let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
        let bias_ref = module.bias.as_ref().unwrap();
        let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);

        let mut mtree = ModuleTree::build(&module);

        assert_eq!(
            format!("{:#?}", mtree),
            indoc::formatdoc! {r#"
                ModuleTree {{
                  <ModuleTree version="{MODULE_TREE_VERSION}">
                    <Structure>
                      <Linear id="n:1" class="struct">
                        <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
                        <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
                      </Linear>
                    </Structure>
                  </ModuleTree>
                }}"#,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );

        assert_eq!(
            format!("{:?}", mtree),
            indoc::formatdoc! {r#"ModuleTree {{<ModuleTree version="{MODULE_TREE_VERSION}"><Structure><Linear id="n:1" class="struct"><Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/><Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/></Linear></Structure></ModuleTree>}}"#,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_to_xml_cuda() {
        test_to_xml::<burn::backend::Cuda>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_to_xml_wgpu() {
        test_to_xml::<burn::backend::Wgpu>();
    }

    fn test_to_xml<B: Backend>() {
        let device = Default::default();
        let module: Linear<B> = LinearConfig::new(2, 3).init(&device);

        let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
        let bias_ref = module.bias.as_ref().unwrap();
        let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);

        let mut mtree = ModuleTree::build(&module);

        assert_eq!(
            mtree.to_xml(true),
            indoc::formatdoc! {r#"
                <ModuleTree version="{MODULE_TREE_VERSION}">
                  <Structure>
                    <Linear id="n:1" class="struct">
                      <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
                      <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
                    </Linear>
                  </Structure>
                </ModuleTree>
                "#,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );

        assert_eq!(
            mtree.to_xml(false),
            indoc::formatdoc! {r#"<ModuleTree version="{MODULE_TREE_VERSION}"><Structure><Linear id="n:1" class="struct"><Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/><Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/></Linear></Structure></ModuleTree>"#,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );
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
