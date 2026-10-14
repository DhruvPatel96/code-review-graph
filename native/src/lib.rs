use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use tree_sitter::{Node, Parser};

#[derive(Serialize)]
struct Definition {
    kind: &'static str,
    name: String,
    parent_name: Option<String>,
    params: Option<String>,
    return_type: Option<String>,
    decorators: Vec<String>,
    bases: Vec<String>,
    module_scope: bool,
    line_start: usize,
    line_end: usize,
}

#[derive(Serialize)]
struct ParseResult {
    path: String,
    sha256: String,
    syntax_errors: usize,
    definitions: Vec<Definition>,
    edges: Vec<Edge>,
    imports: Vec<Import>,
    calls: Vec<Call>,
}

#[derive(Serialize)]
struct Edge {
    kind: &'static str,
    source: String,
    target: String,
    line: usize,
    extra: BTreeMap<&'static str, String>,
}

#[derive(Serialize)]
struct Import {
    module: String,
    names: Vec<String>,
    bindings: BTreeMap<String, String>,
    is_from: bool,
    line: usize,
}

#[derive(Serialize)]
struct Call {
    name: String,
    receiver: Option<String>,
    receiver_form: Option<&'static str>,
    enclosing_class: Option<String>,
    enclosing_function: Option<String>,
    line: usize,
    column: usize,
}

#[derive(Default)]
struct Extraction {
    definitions: Vec<Definition>,
    imports: Vec<Import>,
    calls: Vec<Call>,
    syntax_errors: usize,
}

fn node_text(node: Node<'_>, source: &[u8]) -> Option<String> {
    node.utf8_text(source).ok().map(ToOwned::to_owned)
}

fn visit(
    node: Node<'_>,
    source: &[u8],
    parent: Option<&str>,
    enclosing_function: Option<&str>,
    decorators: &[String],
    extraction: &mut Extraction,
) {
    if node.is_error() || node.is_missing() {
        extraction.syntax_errors += 1;
    }

    if node.kind() == "decorated_definition" {
        let mut own_decorators = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "decorator" {
                if let Some(text) = node_text(child, source) {
                    own_decorators.push(text.trim().trim_start_matches('@').to_owned());
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "decorator" {
                visit(child, source, parent, enclosing_function, &[], extraction);
            } else if matches!(child.kind(), "class_definition" | "function_definition") {
                visit(
                    child,
                    source,
                    parent,
                    enclosing_function,
                    &own_decorators,
                    extraction,
                );
            }
        }
        return;
    }

    if node.kind() == "call" {
        if let Some(callee) = node.child_by_field_name("function") {
            let (name, receiver, receiver_form) = match callee.kind() {
                "identifier" => (node_text(callee, source), None, None),
                "attribute" => {
                    let method = callee
                        .child_by_field_name("attribute")
                        .and_then(|part| node_text(part, source));
                    let object = callee.child_by_field_name("object");
                    let receiver = object.and_then(|part| match part.kind() {
                        "identifier" => node_text(part, source),
                        "attribute" => {
                            let root = part.child_by_field_name("object")?;
                            let field = part.child_by_field_name("attribute")?;
                            if node_text(root, source)?.as_str() == "self" {
                                node_text(field, source)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    });
                    let form = if receiver.is_none() {
                        Some("expression")
                    } else {
                        None
                    };
                    (method, receiver, form)
                }
                _ => (None, None, None),
            };
            if let Some(name) = name {
                extraction.calls.push(Call {
                    name,
                    receiver,
                    receiver_form,
                    enclosing_class: parent.map(ToOwned::to_owned),
                    enclosing_function: enclosing_function.map(ToOwned::to_owned),
                    line: node.start_position().row + 1,
                    column: node.start_position().column,
                });
            }
        }
    }

    if node.kind() == "import_statement" {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            let module_node = if child.kind() == "aliased_import" {
                child.child_by_field_name("name")
            } else {
                Some(child)
            };
            if let Some(module) = module_node.and_then(|part| node_text(part, source)) {
                extraction.imports.push(Import {
                    module,
                    names: Vec::new(),
                    bindings: BTreeMap::new(),
                    is_from: false,
                    line: node.start_position().row + 1,
                });
            }
        }
    } else if node.kind() == "import_from_statement" {
        if let Some(module_node) = node.child_by_field_name("module_name") {
            if let Some(module) = node_text(module_node, source) {
                let mut names = Vec::new();
                let mut bindings = BTreeMap::new();
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    if child.id() == module_node.id() {
                        continue;
                    }
                    match child.kind() {
                        "dotted_name" | "identifier" => {
                            if let Some(name) = node_text(child, source) {
                                names.push(name.clone());
                                bindings.insert(name.clone(), name);
                            }
                        }
                        "aliased_import" => {
                            if let Some(name) = child
                                .child_by_field_name("name")
                                .and_then(|part| node_text(part, source))
                            {
                                let alias = child
                                    .child_by_field_name("alias")
                                    .and_then(|part| node_text(part, source))
                                    .unwrap_or_else(|| name.clone());
                                names.push(name.clone());
                                bindings.insert(alias, name);
                            }
                        }
                        _ => {}
                    }
                }
                extraction.imports.push(Import {
                    module,
                    names,
                    bindings,
                    is_from: true,
                    line: node.start_position().row + 1,
                });
            }
        }
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
    let next_function_owned = match kind {
        Some("Function") => name.clone(),
        Some("Class") => None,
        _ => enclosing_function.map(ToOwned::to_owned),
    };
    if let (Some(kind), Some(name)) = (kind, name) {
        let start = node.start_position();
        let end = node.end_position();
        extraction.definitions.push(Definition {
            kind,
            name,
            parent_name: parent.map(ToOwned::to_owned),
            params: node
                .child_by_field_name("parameters")
                .and_then(|child| node_text(child, source)),
            return_type: node
                .child_by_field_name("return_type")
                .and_then(|child| node_text(child, source)),
            decorators: decorators.to_vec(),
            bases: if kind == "Class" {
                node.child_by_field_name("superclasses")
                    .map(|arguments| {
                        let mut cursor = arguments.walk();
                        arguments
                            .named_children(&mut cursor)
                            .filter(|child| matches!(child.kind(), "identifier" | "attribute"))
                            .filter_map(|child| node_text(child, source))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            },
            module_scope: parent.is_none() && enclosing_function.is_none(),
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
            next_function_owned.as_deref(),
            &[],
            extraction,
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
    let mut extraction = Extraction::default();
    visit(tree.root_node(), &source, None, None, &[], &mut extraction);
    let Extraction {
        mut definitions,
        imports,
        calls,
        syntax_errors,
    } = extraction;
    definitions.sort_by(|left, right| {
        (left.line_start, left.line_end, &left.kind, &left.name).cmp(&(
            right.line_start,
            right.line_end,
            &right.kind,
            &right.name,
        ))
    });
    let edges = definitions
        .iter()
        .flat_map(|definition| {
            let (source, target) = match (&definition.parent_name, definition.kind) {
                (Some(parent), "Function") => (
                    format!("{path}::{parent}"),
                    format!("{path}::{parent}.{}", definition.name),
                ),
                (Some(parent), _) => (
                    path.clone(),
                    format!("{path}::{parent}.{}", definition.name),
                ),
                (None, _) => (path.clone(), format!("{path}::{}", definition.name)),
            };
            let mut edges = vec![Edge {
                kind: "CONTAINS",
                source,
                target,
                line: definition.line_start,
                extra: BTreeMap::new(),
            }];
            if definition.kind == "Class" {
                let class_name = match &definition.parent_name {
                    Some(parent) => format!("{path}::{parent}.{}", definition.name),
                    None => format!("{path}::{}", definition.name),
                };
                edges.extend(definition.bases.iter().map(|base| Edge {
                    kind: "INHERITS",
                    source: class_name.clone(),
                    target: base.clone(),
                    line: definition.line_start,
                    extra: BTreeMap::new(),
                }));
            }
            edges
        })
        .collect();
    Ok(ParseResult {
        path,
        sha256: format!("{:x}", Sha256::digest(&source)),
        syntax_errors,
        definitions,
        edges,
        imports,
        calls,
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
