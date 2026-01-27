//! Exact Match Union Patterns

use crate::regex::regex_wrapper::RegexWrapperPattern;
use alloc::format;
use alloc::vec::Vec;

/// Create a union pattern of exact matches.
///
/// This will always be a [`RegexWrapperPattern::Basic`] variant.
pub fn exact_match_union_regex_pattern<S: AsRef<str>>(alts: &[S]) -> RegexWrapperPattern {
    let parts = alts
        .iter()
        .map(|s| fancy_regex::escape(s.as_ref()))
        .collect::<Vec<_>>();
    RegexWrapperPattern::Basic(format!("({})", parts.join("|")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex::RegexWrapper;
    use alloc::vec;

    #[test]
    fn test_fixed_alternative_list() {
        let alternatives = ["apple", "[x]", "boat"];

        let pattern = exact_match_union_regex_pattern(&alternatives);
        assert_eq!(pattern.as_str(), r"(apple|\[x\]|boat)");

        let re: RegexWrapper = exact_match_union_regex_pattern(&alternatives)
            .compile()
            .unwrap();

        let text = "apple 123 [x] xyz boat";
        assert_eq!(re.find_iter(text).count(), 3);

        assert_eq!(
            re.find_iter(text).map(|m| m.range()).collect::<Vec<_>>(),
            vec![0..5, 10..13, 18..22]
        );
    }
}
