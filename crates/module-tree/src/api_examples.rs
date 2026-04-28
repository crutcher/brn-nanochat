#[cfg(test)]
#[allow(unused)]
mod tests {
    use std::collections::HashSet;

    use burn::{
        module::{
            Module,
            ParamId,
        },
        nn::{
            Linear,
            LinearConfig,
            LinearLayout,
        },
        prelude::Backend,
        tensor::Shape,
    };
    use zsl_chat::gpt::gpt_model::{
        GPT,
        GPTConfig,
    };

    use crate::{
        ModuleTree,
        ModuleTreeQuery,
        burn_ext::burn_desc::{
            ParamDesc,
            TensorDesc,
            TensorKindDesc,
            TensorParamDesc,
        },
        error::BunsenResult,
    };

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_tensor_param_desc_example() -> BunsenResult<()> {
        tensor_param_desc_example::<burn::backend::Cuda>(&Default::default())
    }

    /// This is an example of the [`TensorParamDesc`] API.
    fn tensor_param_desc_example<B: Backend>(device: &B::Device) -> BunsenResult<()> {
        // Create a Linear module, with a bias:
        // * `weight` - `Param<Tensor<B, 2>>` [d_input, d_output].
        // * `bias` - `Option<Param<Tensor<B, 1>>>` [d_output].
        let d_input = 2;
        let d_output = 3;
        let module: Linear<B> = LinearConfig::new(d_input, d_output).init(device);

        // [`TensorParamDesc`] can describe a `Param<Tensor<B, R, K>>`:
        let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
        let bias_ref = module.bias.as_ref().unwrap();
        let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);

        // [`TensorParamDesc`] exposes the basic `Param` and `Tensor` metadata:
        assert_eq!(weight_desc.param_id(), module.weight.id);

        // [`TensorKindDesc`] is an enum which describes the current kind variants:
        assert_eq!(weight_desc.kind(), TensorKindDesc::Float);
        assert_eq!(weight_desc.dtype(), module.weight.dtype());

        assert_eq!(weight_desc.shape(), &module.weight.shape());
        assert_eq!(weight_desc.shape(), &Shape::new([d_input, d_output]));

        // [`TensorParamDesc`] also provides some convience methods:
        assert_eq!(weight_desc.rank(), 2);
        assert_eq!(
            weight_desc.size_estimate(),
            module.weight.dtype().size() * 2 * 3
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_linear_module_tree_example() -> BunsenResult<()> {
        linear_module_tree_example::<burn::backend::Cuda>(&Default::default())
    }

    fn linear_module_tree_example<B: Backend>(device: &B::Device) -> BunsenResult<()> {
        // Create a Linear module, with a bias:
        // * `weight` - `Param<Tensor<B, 2>>` [d_input, d_output].
        // * `bias` - `Option<Param<Tensor<B, 1>>>` [d_output].
        let d_input = 2;
        let d_output = 3;
        let module: Linear<B> = LinearConfig::new(d_input, d_output).init(device);

        let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
        let bias_desc: TensorParamDesc = TensorParamDesc::from(module.bias.as_ref().unwrap());

        // Build a ModuleTree from the module.
        // As the ModuleTree holds non-Send active active query environment,
        // it must be `mut` to be useful.
        let mut mtree = ModuleTree::build(&module);

        assert_eq!(
            &mtree.to_xml(),
            indoc::formatdoc! {r#"
              <ModuleTree version="{}">
            "#,

            }
        );

        // [`ModuleTree::param_ids`] iterates over all [`ParamId`]s.
        //
        // This is a useful way to get all the parameter ids in a module;
        // but it is actually a wrapper over a series of more complex steps.
        //
        //   mtree
        //     // Create a query over the whole structure.
        //     .query()
        //     // Sub-select all `Param` nodes.
        //     .params()
        //     // Extract all `@param_id` attributes to `ParamId`s.
        //     .to_param_ids()?
        //     // Collect the results into a `Vec`.
        //     .collect::<Vec<_>>();
        //
        // Which further expands to:
        //
        //   mtree
        //     // Create a query over the whole structure.
        //     .query()
        //     // Sub-select all `Param` nodes.
        //     .select("descedant-or-self::Param")
        //     // Extract all `@param_id` attributes to `ParamId`s.
        //     .to_param_ids()?
        //     // Collect the results into a `Vec`.
        //     .collect::<Vec<_>>();
        let ids: Vec<ParamId> = mtree.param_ids()?.collect();

        // IMPORTANT: Module Tree Ordering
        //
        // `burn` Modules order their children in a stable and specific order,
        // determined by the order of their declaration in the source code,
        // and the current semantics of the `Module` derive macro.
        //
        // Where possible, you should not rely upon this; and should prefer
        // to use `HashSet<ParamId>` or similar to shield yourself from
        // ordering variation; particularly as you'll generally be using
        // this machinery when doing subset calculations.
        assert_eq!(ids, [module.weight.id, module.bias.as_ref().unwrap().id]);

        // [`ModuleTree::param_descs`] iterates over descriptions of every parameter.
        //
        // This leverages the [`TensorParamDesc`] API to strip generics from
        // the introspection api.
        //
        // Similar to [`ModuleTree::param_ids`], this is a wrapper over a series of more
        // complex steps.
        //
        //   mtree
        //     // Create a query over the whole structure.
        //     .query()
        //     // Sub-select all `Param` nodes.
        //     .params()
        //     // Build a `TensorParamDesc` for each `Param`.
        //     .to_param_descs()?
        //     // Collect the results into a `Vec`.
        //     .collect::<Vec<_>>();
        //
        // Which further expands to:
        //
        //   mtree
        //     // Create a query over the whole structure.
        //     .query()
        //     // Sub-select all `Param` nodes.
        //     .select("descedant-or-self::Param")
        //     // Build a `TensorParamDesc` for each `Param`.
        //     .to_param_descs()?
        //     // Collect the results into a `Vec`.
        //     .collect::<Vec<_>>();
        let descs: Vec<TensorParamDesc> = mtree.param_descs()?.collect();

        Ok(())
    }

    #[derive(Module, Debug)]
    struct BModule<B: Backend> {
        a: Linear<B>,
        b: Linear<B>,
    }

    #[derive(Module, Debug)]
    enum ExampleEnumModule<B: Backend> {
        Foo(Linear<B>),

        Bar(BModule<B>),
    }

    #[derive(Module, Debug)]
    struct ExampleStructModule<B: Backend> {
        seq: Vec<Linear<B>>,
        tup: (Linear<B>, Linear<B>),
        arr: [Linear<B>; 1],
    }

    impl<B: Backend> ExampleStructModule<B> {
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
