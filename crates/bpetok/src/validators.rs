//! Validators for various configuration options.
use fancy_regex::Regex;

/// The size of the u8 space.
pub const U8_SIZE: usize = 256;

/// Validates and returns the vocabulary size, ensuring it's at least the size of the u8 space.
pub fn try_vocab_size(vocab_size: usize) -> anyhow::Result<usize> {
    if vocab_size < U8_SIZE {
        Err(anyhow::anyhow!(
            "vocab_size ({vocab_size}) must be >= 256 (the size of the u8 space)"
        ))
    } else {
        Ok(vocab_size)
    }
}

/// Validates and returns the vocab size, panicking if it's too small.
pub fn expect_vocab_size(vocab_size: usize) -> usize {
    try_vocab_size(vocab_size).unwrap()
}

/// Validates and returns a regex pattern, panicking if it fails to compile.
pub fn try_regex(pattern: &str) -> anyhow::Result<Regex> {
    Regex::new(pattern)
        .map_err(|_| anyhow::anyhow!("regex pattern compilation failed: {}", pattern))
}

/// Validates and returns a regex pattern, panicking if it fails to compile.
pub fn expect_regex<S: AsRef<str>>(pattern: S) -> Regex {
    try_regex(pattern.as_ref()).unwrap()
}

/// Validates and returns parallel processing options.
pub fn try_parallel(parallel: bool) -> anyhow::Result<bool> {
    #[cfg(not(feature = "rayon"))]
    if parallel {
        return Err(anyhow::anyhow!(
            "Parallel processing requires the `rayon` feature to be enabled."
        ));
    }
    Ok(parallel)
}

/// Validates and returns parallel processing options, panicking if it's not enabled.
pub fn expect_parallel(parallel: bool) -> bool {
    try_parallel(parallel).unwrap()
}
