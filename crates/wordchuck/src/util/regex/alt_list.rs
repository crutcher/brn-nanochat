//! Alternative List Utilities

use crate::util::regex::regex_wrapper::RegexWrapperPattern;

/// Create a regex pattern for a fixed list of alternatives.
pub fn fixed_alternative_list_regex<S: AsRef<str>>(alts: &[S]) -> RegexWrapperPattern {
    let parts = alts
        .iter()
        .map(|s| fancy_regex::escape(s.as_ref()))
        .collect::<Vec<_>>();
    RegexWrapperPattern::Basic(format!("({})", parts.join("|")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_alternative_list_regex() {
        let pattern = fixed_alternative_list_regex(&["apple", "[x]", "boat"]);
        assert_eq!(pattern.as_str(), r"(apple|\[x\]|boat)");
    }
}
