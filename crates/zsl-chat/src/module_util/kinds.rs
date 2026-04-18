/// Encodes the kind of a Module Parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ParamKind {
    /// A Bool Parameter.
    Bool,

    /// A Float Parameter.
    Float,

    /// An Int Parameter.
    Int,
}
