#![recursion_limit = "512"]

pub mod burn_ext;
mod module_tree;
mod type_util;
mod xot_util;

pub mod burn_enc;
pub mod constants;
pub mod error;
pub(crate) mod implementation;
pub mod xee_util;

#[cfg(test)]
pub mod api_examples;

#[doc(inline)]
pub use module_tree::*;
#[doc(inline)]
pub use xot_util::*;
