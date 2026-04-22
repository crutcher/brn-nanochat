use std::collections::BTreeMap;

use burn::{
    module::Module,
    prelude::Backend,
};

use crate::param_map::{
    ParamDesc,
    ParamPath,
    builder::ParamMapBuildingVisitor,
};

/// A map from module paths to parameter kinds.
#[derive(Debug, Clone, Default)]
pub struct ParamMap {
    params: BTreeMap<ParamPath, ParamDesc>,
}

impl ParamMap {
    /// Collects the parameter map from a module.
    pub fn collect<M: Module<B>, B: Backend>(module: &M) -> Self {
        let mut visitor = ParamMapBuildingVisitor::<B>::default();
        module.visit(&mut visitor);
        visitor.param_map()
    }

    /// Adds a parameter to the map.
    pub fn add_param(
        &mut self,
        desc: ParamDesc,
    ) {
        self.params.insert(desc.path.clone(), desc);
    }

    /// Returns an iterator over the parameter map.
    pub fn iter(&self) -> impl Iterator<Item = (&ParamPath, &ParamDesc)> {
        self.params.iter()
    }

    /// Returns the number of parameters in the map.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        nn::{
            Linear,
            LinearConfig,
        },
        tensor::DType,
    };

    use super::*;
    use crate::{
        ParamKind,
        param_map::ParamTag,
    };

    #[test]
    fn test_param_kind() {
        assert_eq!(ParamKind::Bool, ParamKind::Bool);
        assert_ne!(ParamKind::Bool, ParamKind::Float);
    }

    #[test]
    fn test_param_ref() {
        let ref1 = ParamTag::new(1.into(), ParamKind::Bool, DType::Bool, [2, 3].into());
        let ref1_dup = ParamTag::new(1.into(), ParamKind::Bool, DType::Bool, [2, 3].into());
        let ref1_cp = ref1.clone();

        assert_eq!(ref1, ref1_dup);
        assert_eq!(ref1, ref1_cp);

        assert_eq!(ref1.id(), 1.into());
        assert_eq!(ref1.kind(), ParamKind::Bool);

        let ref2 = ParamTag::new(2.into(), ParamKind::Float, DType::F32, [2, 3].into());
        let ref3 = ParamTag::new(3.into(), ParamKind::Int, DType::I32, [2, 3].into());

        assert_eq!(ref2.id(), 2.into());
        assert_eq!(ref2.kind(), ParamKind::Float);

        assert_eq!(ref3.id(), 3.into());
        assert_eq!(ref3.kind(), ParamKind::Int);

        assert_ne!(ref1, ref2);
        assert_ne!(ref1, ref3);
    }

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        seq: Vec<Linear<B>>,
    }

    impl<B: Backend> TestModule<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                seq: vec![LinearConfig::new(10, 10).init(device)],
            }
        }
    }

    #[test]
    fn test_module_path() {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let _param_map = ParamMap::collect(&module);

        /*
        assert_eq!(
            &param_map.iter().collect::<Vec<_>>(),
            &vec![
                (
                    &ParamPath(vec![
                        ParamPathNode::new("seq", "Struct:TestModule"),
                        ParamPathNode::new("0", "Vec"),
                        ParamPathNode::new("bias", "Struct:Linear"),
                    ]),
                    &ParamTag::new(
                        module.seq[0].bias.as_ref().unwrap().id,
                        ParamKind::Float,
                        <B as Backend>::FloatElem::dtype(),
                        [10].into()
                    )
                ),
                (
                    &ParamPath(vec![
                        ParamPathNode::new("seq", "Struct:TestModule"),
                        ParamPathNode::new("0", "Vec"),
                        ParamPathNode::new("weight", "Struct:Linear"),
                    ]),
                    &ParamTag::new(
                        module.seq[0].weight.id,
                        ParamKind::Float,
                        <B as Backend>::FloatElem::dtype(),
                        [10, 10].into()
                    )
                ),
            ]
        );

         */
    }
}
