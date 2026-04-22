/// Represents a node in a module tree path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamPathNode {
    /// The name of the node.
    name: String,

    /// The name of the container type of the node.
    container: String,
}

impl ParamPathNode {
    /// Creates a new `ModulePathNode`.
    pub fn new(
        name: &str,
        container: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            container: container.to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn container(&self) -> &str {
        &self.container
    }
}

/// Represents a path in a module tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamPath(Vec<ParamPathNode>);

impl ParamPath {
    pub fn new(nodes: Vec<ParamPathNode>) -> Self {
        assert!(!nodes.is_empty());
        Self(nodes)
    }

    pub fn nodes(&self) -> &[ParamPathNode] {
        &self.0
    }

    pub fn push(
        &mut self,
        node: ParamPathNode,
    ) {
        self.0.push(node);
    }

    pub fn path_str(&self) -> String {
        self.0
            .iter()
            .map(|n| n.name.clone())
            .collect::<Vec<_>>()
            .join(".")
    }
}
