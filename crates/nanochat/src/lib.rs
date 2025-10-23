//! # Nanochat Lib

pub mod burn_ext;
pub mod model;

pub fn xyzzy() -> usize {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(xyzzy(), 42);
    }
}
