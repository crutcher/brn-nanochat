use burn::tensor;
/// Encodes the kind of a Module Parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TensorKindDesc {
    /// A Bool Parameter.
    Bool,

    /// A Float Parameter.
    Float,

    /// An Int Parameter.
    Int,
}

/// A trait that binds a burn Tensor Kind to a `ParamKind`.
pub trait ParamKindBinding {
    const KIND: TensorKindDesc;
}

impl ParamKindBinding for tensor::Bool {
    const KIND: TensorKindDesc = TensorKindDesc::Bool;
}

impl ParamKindBinding for tensor::Float {
    const KIND: TensorKindDesc = TensorKindDesc::Float;
}

impl ParamKindBinding for tensor::Int {
    const KIND: TensorKindDesc = TensorKindDesc::Int;
}

impl TensorKindDesc {
    pub const fn for_kind<K: ParamKindBinding>() -> Self {
        K::KIND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_kind() {
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Bool>(),
            TensorKindDesc::Bool
        );
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Float>(),
            TensorKindDesc::Float
        );
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Int>(),
            TensorKindDesc::Int
        );
    }
}
