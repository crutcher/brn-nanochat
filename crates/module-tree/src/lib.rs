#![cfg_attr(feature = "wgpu", recursion_limit = "512")]

#[cfg(test)]
pub mod api_examples;

pub mod burn_ext;
pub mod errors;
pub mod module_visitors;

pub mod xml_support;

mod module_tree;
#[doc(inline)]
pub use module_tree::*;
