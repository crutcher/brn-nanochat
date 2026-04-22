#[cfg(test)]
#[allow(unused)]
mod tests {
    use xee_xpath::{
        Documents,
        Queries,
        Query,
    };
    use xot::{
        Node,
        Xot,
    };

    fn pretty_print(
        xot: &Xot,
        node: Node,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut output_params = xot::output::xml::Parameters {
            indentation: Some(Default::default()),
            ..Default::default()
        };
        println!("{}", xot.serialize_xml_string(output_params, node)?);

        Ok(())
    }

    #[test]
    fn test_xot_construction() -> Result<(), Box<dyn std::error::Error>> {
        let mut xot = Xot::new();

        let root = xot.parse("<p>Example</p>")?;
        let doc_el = xot.document_element(root)?;
        let txt = xot.first_child(doc_el).unwrap();
        let txt_value = xot.text_mut(txt).unwrap();
        txt_value.set("Hello, world!");

        assert_eq!(xot.to_string(root)?, "<p>Hello, world!</p>");

        pretty_print(&xot, root)?;

        Ok(())
    }

    #[test]
    fn test_xee_xpath() -> Result<(), Box<dyn std::error::Error>> {
        // create a new documents object
        let mut documents = Documents::new();
        // load a document from a string
        let doc = documents
            .add_string(
                "http://example.com".try_into().unwrap(),
                "<root><bar>foo</bar></root>",
            )
            .unwrap();

        // create a new queries object
        let queries = Queries::default();

        // create a query expecting a single value in the result sequence
        // try to convert this value into a Rust `String`
        let q = queries.one("/root/string()", |_, item| {
            Ok(item.try_into_value::<String>()?)
        })?;

        // when we execute the query, we need to pass a mutable reference to the
        // documents, and the item against which we want to query. We can also
        // pass in a document handle, as we do here
        let r = q.execute(&mut documents, doc)?;
        assert_eq!(r, "foo");

        pretty_print(documents.xot(), documents.document_node(doc).unwrap())?;

        Ok(())
    }
}
