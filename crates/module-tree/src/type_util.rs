const SEQUENCE_TYPES: &[&str] = &["Vec", "Array", "Tuple"];

pub fn type_is_sequence(name: &str) -> bool {
    SEQUENCE_TYPES.contains(&name)
}

pub fn parse_container_type(container_type: &str) -> (String, String) {
    if let Some((cls, name)) = container_type.split_once(':') {
        let cls = cls.to_lowercase();
        (cls, name.to_string())
    } else {
        ("builtin".to_string(), container_type.to_string())
    }
}
