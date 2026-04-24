#![recursion_limit = "512"]

pub mod burn_ext;
mod mtree;
mod type_util;
mod xot_util;

pub(crate) mod implementation;

#[doc(inline)]
pub use mtree::*;
#[doc(inline)]
pub use xot_util::*;
