#[cfg(test)]
#[allow(unused)]
mod tests {
    use burn::{
        backend::Wgpu,
        module::{
            Module,
            ParamId,
        },
        nn::{
            Linear,
            LinearConfig,
        },
        prelude::Backend,
    };
    use zsl_chat::gpt::gpt_model::{
        GPT,
        GPTConfig,
    };

    use crate::{
        ModuleTree,
        burn_ext::burn_desc::{
            ParamDesc,
            TensorDesc,
        },
        error::BunsenResult,
    };

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
