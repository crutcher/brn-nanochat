//! # Patterns

use crate::regex::re_wrapper::ConstRegexWrapperPattern;
macro_rules! join {
    ($sep:literal, $first:literal $(, $rest:literal)*) => {
        concat!($first $(, $sep, $rest)*)
    };
}

macro_rules! join_pattern {
    ($($e:expr),* $(,)?) => { join!("|", $($e),*) };
}

/// The GPT-2 r50k word pattern.
///
/// Faster than [`GPT2_SLOW_WORD_PATTERN`], optimized for performance.
pub const GPT2_R50K_WORD_PATTERN: ConstRegexWrapperPattern =
    ConstRegexWrapperPattern::Fancy(join_pattern!(
        r"'(?:[sdmt]|ll|ve|re)",
        r" ?\p{L}++",
        r" ?\p{N}++",
        r" ?[^\s\p{L}\p{N}]++",
        r"\s++$",
        r"\s+(?!\S)",
        r"\s",
    ));

/// The original GPT-2 word pattern.
pub const GPT2_SLOW_WORD_PATTERN: ConstRegexWrapperPattern =
    ConstRegexWrapperPattern::Fancy(join_pattern!(
        r"'s",
        r"'t",
        r"'re",
        r"'ve",
        r"'m",
        r"'ll",
        r"'d",
        r" ?[\p{L}]+",
        r" ?[\p{N}]+",
        r" ?[^\s\p{L}\p{N}]+",
        r"\s+(?!\S)",
        r"\s+",
    ));

/// The GPT-3 cl100K word pattern.
pub const GPT3_CL100K_WORD_PATTERN: ConstRegexWrapperPattern =
    ConstRegexWrapperPattern::Fancy(join_pattern!(
        r"'(?i:[sdmt]|ll|ve|re)",
        r"[^\r\n\p{L}\p{N}]?+\p{L}++",
        r"\p{N}{1,3}+",
        r" ?[^\s\p{L}\p{N}]++[\r\n]*+",
        r"\s++$",
        r"\s*[\r\n]",
        r"\s+(?!\S)",
        r"\s",
    ));

/// The GPT-5 o220k word pattern.
pub const GPT5_O220K_WORD_PATTERN: ConstRegexWrapperPattern = ConstRegexWrapperPattern::Fancy(
    join_pattern!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+"
    ),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        assert!(GPT2_R50K_WORD_PATTERN.compile().is_ok());
        assert!(GPT2_SLOW_WORD_PATTERN.compile().is_ok());

        assert!(GPT3_CL100K_WORD_PATTERN.compile().is_ok());

        assert!(GPT3_CL100K_WORD_PATTERN.compile().is_ok());
        assert!(GPT5_O220K_WORD_PATTERN.compile().is_ok());
    }
}
