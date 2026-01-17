//! Alternative List Utilities

use crate::util::regex::re_wrapper::{RegexWrapper, RegexWrapperPattern};

/// Create a list of fixed-match alternatives into a regex pattern.
///
/// This will always be a [`RegexWrapperPattern::Basic`] variant.
pub fn fixed_alternative_list_regex_pattern<S: AsRef<str>>(alts: &[S]) -> RegexWrapperPattern {
    let parts = alts
        .iter()
        .map(|s| fancy_regex::escape(s.as_ref()))
        .collect::<Vec<_>>();
    RegexWrapperPattern::Basic(format!("({})", parts.join("|")))
}

/// Create a list of fixed-match alternatives into a compiled regex.
pub fn fixed_alternative_list_regex_wrapper<S: AsRef<str>>(alts: &[S]) -> RegexWrapper {
    fixed_alternative_list_regex_pattern(alts)
        .compile()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_alternative_list() {
        let alternatives = ["apple", "[x]", "boat"];

        let pattern = fixed_alternative_list_regex_pattern(&alternatives);
        assert_eq!(pattern.as_str(), r"(apple|\[x\]|boat)");

        let re = fixed_alternative_list_regex_wrapper(&alternatives);

        let text = "apple 123 [x] xyz boat";
        assert_eq!(re.find_iter(text).count(), 3);

        assert_eq!(
            re.find_iter(text).map(|m| m.range()).collect::<Vec<_>>(),
            vec![0..5, 10..13, 18..22]
        );
    }
}
