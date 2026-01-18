//! # Regex Utilities
//!
//! This module attempts to balance two problems:
//! * Pattern Complexity
//! * Concurrence Contention
//!
//! ### Pattern Complexity
//!
//! A number of popular in-use LLM Tokenizer Regex Patterns require extended regex
//! machinery provided by the [`fancy_regex`] crate; but naturally, this has performance
//! costs. We'd prefer to avoid using the [`fancy_regex`] crate when possible, falling back
//! on the standard [`regex`] crate when patterns permit this.
//!
//! This recurses into two problems:
//!
//! * Labeling Patterns - [`RegexWrapperPattern`]
//!   * [`RegexWrapperPattern::Basic`] - a pattern which was written for [`regex`].
//!   * [`RegexWrapperPattern::Fancy`] - a pattern which was written for [`fancy_regex`].
//!   * [`RegexWrapperPattern::Adaptive`] - unknown target, try basic; then fall-up to fancy.
//! * Wrapping Compiled Regex - [`RegexWrapper`]
//!
//! The [`RegexWrapper`] type supports only one operation, ``find_iter()`` which requires
//! some adaptation of the `Iterator` stream to function.
//!
//! ### Concurrence Contention
//!
//! There are some observed thread contentions deep in compiled regex objects,
//! short fights over shared internal buffers. In high parallelism, heavy-regex workloads,
//! this can have a large performance impact.
//!
//! At the same time, per-thread local data structures, locks, and cloning introduce
//! dependencies which may not be appropriate in all environments.
//!
//! The chosen solution to this is the combination of:
//! * [`RegexSupplier`] / [`RegexSupplierHandle`]
//! * [`parallel_regex_supplier`].
//!
//! Users of a [`RegexWrapper`] that *may* be under heavy thread contention should use
//! [`parallel_regex_supplier`]; which in some build environments will provide
//! a thread local clone regex supplier, and in some, a simple clone implementation.

mod alt_list;
mod re_supplier;
mod re_wrapper;

pub use alt_list::{fixed_alternative_list_regex_pattern, fixed_alternative_list_regex_wrapper};
pub use re_supplier::{RegexSupplier, RegexSupplierHandle, SimpleRegexSupplier};
pub use re_wrapper::{ErrorWrapper, RegexWrapper, RegexWrapperHandle, RegexWrapperPattern};

#[cfg(feature = "std")]
mod re_pool;
#[cfg(feature = "std")]
use re_pool::RegexWrapperPool;

/// Build a regex supplier for (potentially) parallel execution.
///
/// Users of a [`RegexWrapper`] that *may* be under heavy thread contention should use
/// [`parallel_regex_supplier`]; which in some build environments will provide
/// a thread local clone regex supplier, and in some, a simple clone implementation.
pub fn parallel_regex_supplier<R>(regex: R) -> RegexSupplierHandle
where
    R: Into<RegexWrapperHandle>,
{
    let regex = regex.into();

    #[cfg(feature = "std")]
    return alloc::sync::Arc::new(RegexWrapperPool::new(regex));

    #[cfg(not(feature = "std"))]
    return regex.into();
}
