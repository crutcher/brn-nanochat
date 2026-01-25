//! Exact Match Union Patterns

use crate::regex::regex_wrapper::{RegexWrapper, RegexWrapperPattern};

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

/// Create a union pattern of exact matches, compiled into a [`RegexWrapper`].
///
/// See: [`exact_match_union_regex_pattern`]
pub fn exact_match_union_regex_wrapper<S: AsRef<str>>(alts: &[S]) -> RegexWrapper {
    exact_match_union_regex_pattern(alts).compile().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_alternative_list() {
        let alternatives = ["apple", "[x]", "boat"];

        let pattern = exact_match_union_regex_pattern(&alternatives);
        assert_eq!(pattern.as_str(), r"(apple|\[x\]|boat)");

        let re = exact_match_union_regex_wrapper(&alternatives);

        let text = "apple 123 [x] xyz boat";
        assert_eq!(re.find_iter(text).count(), 3);

        assert_eq!(
            re.find_iter(text).map(|m| m.range()).collect::<Vec<_>>(),
            vec![0..5, 10..13, 18..22]
        );
    }
}
