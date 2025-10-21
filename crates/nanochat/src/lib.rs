//! # Nanochat Lib

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
