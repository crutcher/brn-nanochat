#![recursion_limit = "512"]

pub mod burn_ext;
mod mtree;
mod type_util;
mod xot_util;

pub mod error;
pub(crate) mod implementation;
pub mod xee_util;

#[doc(inline)]
pub use mtree::*;
#[doc(inline)]
pub use xot_util::*;
