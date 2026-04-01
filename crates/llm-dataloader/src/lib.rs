pub mod cursor;
pub mod loader;

pub fn foo() -> usize {
    42
}

#[cfg(test)]
mod tests {
    use crate::foo;

    #[test]
    fn test_foo() {
        assert_eq!(foo(), 42);
    }
}
