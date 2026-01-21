//! # Regex Supplier Trait

use crate::util::regex::RegexWrapper;
use alloc::sync::Arc;

/// Common Regex Supplier Handle Type
pub type RegexSupplierHandle = Arc<dyn RegexSupplier>;

/// Regex Supplier Trait
pub trait RegexSupplier: Sync + Send {
    /// Get the regex.
    fn get_regex(&self) -> Arc<RegexWrapper>;

    /// Get the regex pattern.
    fn get_pattern(&self) -> String {
        self.get_regex().as_str().to_string()
    }
}

impl core::fmt::Debug for dyn RegexSupplier {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "RegexSupplier({})", self.get_pattern())
    }
}

impl RegexSupplier for RegexWrapper {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        Arc::new(self.clone())
    }
}

impl RegexSupplier for Arc<RegexWrapper> {
    fn get_regex(&self) -> Arc<RegexWrapper> {
        self.clone()
    }
}
