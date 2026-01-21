//! # Patterns

use crate::util::regex::re_wrapper::ConstRegexWrapperPattern;

/// The GPT-4 style regex pattern for splitting text
pub const GPT4_PATTERN: ConstRegexWrapperPattern = ConstRegexWrapperPattern::Fancy(
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        assert!(GPT4_PATTERN.compile().is_ok());
    }
}
