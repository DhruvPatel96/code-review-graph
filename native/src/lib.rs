use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tree_sitter::{Node, Parser};

#[derive(Serialize)]
struct Definition {
    kind: &'static str,
    name: String,
    parent_name: Option<String>,
    line_start: usize,
    line_end: usize,
}

#[derive(Serialize)]
struct ParseResult {
    path: String,
    sha256: String,
    syntax_errors: usize,
    definitions: Vec<Definition>,
}

fn node_text(node: Node<'_>, source: &[u8]) -> Option<String> {
    node.utf8_text(source).ok().map(ToOwned::to_owned)
}

fn visit(
    node: Node<'_>,
    source: &[u8],
    parent: Option<&str>,
    definitions: &mut Vec<Definition>,
    syntax_errors: &mut usize,
) {
    if node.is_error() || node.is_missing() {
        *syntax_errors += 1;
    }

    let kind = match node.kind() {
        "class_definition" => Some("Class"),
        "function_definition" => Some("Function"),
        _ => None,
    };
    let name = kind
        .and_then(|_| node.child_by_field_name("name"))
        .and_then(|name_node| node_text(name_node, source));
    let next_parent = if kind == Some("Class") {
        name.clone()
    } else {
        parent.map(ToOwned::to_owned)
    };
    if let (Some(kind), Some(name)) = (kind, name) {
        let start = node.start_position();
        let end = node.end_position();
        definitions.push(Definition {
            kind,
            name,
            parent_name: parent.map(ToOwned::to_owned),
            line_start: start.row + 1,
            line_end: end.row + 1,
        });
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit(
            child,
            source,
            next_parent.as_deref(),
            definitions,
            syntax_errors,
        );
    }
}

fn parse_one(path: String, source: Vec<u8>) -> Result<ParseResult, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|error| format!("could not load Python grammar: {error}"))?;
    let tree = parser
        .parse(&source, None)
        .ok_or_else(|| "tree-sitter returned no syntax tree".to_owned())?;
    let mut definitions = Vec::new();
    let mut syntax_errors = 0;
    visit(
        tree.root_node(),
        &source,
        None,
        &mut definitions,
        &mut syntax_errors,
    );
    definitions.sort_by(|left, right| {
        (left.line_start, left.line_end, &left.kind, &left.name).cmp(&(
            right.line_start,
            right.line_end,
            &right.kind,
            &right.name,
        ))
    });
    Ok(ParseResult {
        path,
        sha256: format!("{:x}", Sha256::digest(&source)),
        syntax_errors,
        definitions,
    })
}

/// Parse a batch in parallel and cross the Python boundary once.
#[pyfunction]
fn parse_python_batch(items: Vec<(String, Vec<u8>)>) -> PyResult<String> {
    let parsed: Result<Vec<_>, _> = items
        .into_par_iter()
        .map(|(path, source)| parse_one(path, source))
        .collect();
    serde_json::to_string(&parsed.map_err(PyRuntimeError::new_err)?)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parse_python_batch, module)?)?;
    module.add_function(wrap_pyfunction!(core_version, module)?)?;
    Ok(())
}
