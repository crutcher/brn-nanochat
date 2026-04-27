use std::{
    ops::{
        Deref,
        DerefMut,
    },
    sync::{
        Arc,
        Mutex,
        MutexGuard,
    },
};

use xee_xpath::Documents;
use xot::{
    NameId,
    Node,
    Xot,
};

pub fn pretty_print_node(
    xot: &Xot,
    node: Node,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        xot.serialize_xml_string(
            xot::output::xml::Parameters {
                indentation: Some(Default::default()),
                ..Default::default()
            },
            node
        )?
    );

    Ok(())
}

pub struct GuardHandle<'a, T> {
    guard: MutexGuard<'a, T>,
}

impl<'a, T> GuardHandle<'a, T> {
    pub fn new(guard: MutexGuard<'a, T>) -> Self {
        Self { guard }
    }
}

impl<'a, T> Deref for GuardHandle<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<'a, T> DerefMut for GuardHandle<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

#[derive(Debug, Default, Clone)]
pub struct XotDocumentsHandle {
    docs: Arc<Mutex<Documents>>,
}

impl XotDocumentsHandle {
    pub fn new() -> XotDocumentsHandle {
        Self {
            docs: Arc::new(Mutex::new(Documents::new())),
        }
    }

    pub fn lock(&self) -> XotHandle<'_> {
        XotHandle::new(self.docs.lock().unwrap())
    }
}

pub struct XotHandle<'a> {
    docs: GuardHandle<'a, Documents>,
}

impl<'a> XotHandle<'a> {
    pub(crate) fn new(docs: MutexGuard<'a, Documents>) -> Self {
        Self {
            docs: GuardHandle::new(docs),
        }
    }

    pub fn add_name(
        &mut self,
        name: &str,
    ) -> NameId {
        self.xot_mut().add_name(name)
    }

    pub fn docs(&self) -> &Documents {
        &self.docs
    }

    pub fn docs_mut(&mut self) -> &mut Documents {
        &mut self.docs
    }

    pub fn xot(&self) -> &Xot {
        self.docs.xot()
    }

    pub fn xot_mut(&mut self) -> &mut Xot {
        self.docs.deref_mut().xot_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        XotDocumentsHandle,
        error::BunsenResult,
    };

    #[test]
    #[allow(unused)]
    fn test_scratch() -> BunsenResult<()> {
        let w: XotDocumentsHandle = Default::default();
        let mut h = w.lock();

        let mtree_nid = h.add_name("ModuleTree");

        let xot = h.xot_mut();
        let root = xot.new_element(mtree_nid);
        let doc = xot.new_document_with_element(root).unwrap();

        Ok(())
    }
}
