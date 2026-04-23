const SEQUENCE_TYPES: &[&str] = &["Vec", "Array", "Tuple"];

pub fn type_is_sequence(name: &str) -> bool {
    SEQUENCE_TYPES.contains(&name)
}

pub fn type_is_struct(name: &str) -> bool {
    name.starts_with("Struct:")
}
