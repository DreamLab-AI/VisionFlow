//! Markdown processing utilities with JSON conversion and structured parsing

use crate::{Result, UtilsError};
use pulldown_cmark::{Event, Parser, Tag, CodeBlockKind, HeadingLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Configuration for markdown processing
#[derive(Debug, Clone)]
pub struct MarkdownConfig {
    pub extract_code_blocks: bool,
    pub preserve_formatting: bool,
    pub include_metadata: bool,
    pub max_heading_level: u8,
    pub parse_tables: bool,
}

impl Default for MarkdownConfig {
    fn default() -> Self {
        Self {
            extract_code_blocks: true,
            preserve_formatting: false,
            include_metadata: true,
            max_heading_level: 6,
            parse_tables: true,
        }
    }
}

/// Represents a markdown document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownDocument {
    pub metadata: HashMap<String, String>,
    pub title: Option<String>,
    pub sections: Vec<MarkdownSection>,
    pub code_blocks: Vec<CodeBlock>,
    pub tables: Vec<MarkdownTable>,
    pub links: Vec<Link>,
    pub images: Vec<Image>,
}

/// Represents a section in the markdown document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownSection {
    pub level: u8,
    pub title: String,
    pub content: String,
    pub subsections: Vec<MarkdownSection>,
    pub id: String,
}

/// Represents a code block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub code: String,
    pub line_number: usize,
    pub id: String,
}

/// Represents a table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub id: String,
}

/// Represents a link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub text: String,
    pub url: String,
    pub title: Option<String>,
    pub is_reference: bool,
}

/// Represents an image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub alt_text: String,
    pub url: String,
    pub title: Option<String>,
}

/// Parse markdown file and convert to structured JSON
///
/// # Arguments
/// * `markdown_file` - Input markdown file path
/// * `config` - Markdown processing configuration
///
/// # Returns
/// * `Result<MarkdownDocument>` - Parsed markdown document
pub fn markdown_to_json<P: AsRef<Path>>(
    markdown_file: P,
    config: &MarkdownConfig,
) -> Result<MarkdownDocument> {
    let mut file = File::open(markdown_file)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    parse_markdown_content(&content, config)
}

/// Parse markdown content string to structured document
pub fn parse_markdown_content(
    content: &str,
    config: &MarkdownConfig,
) -> Result<MarkdownDocument> {
    let parser = Parser::new(content);
    let mut document = MarkdownDocument {
        metadata: HashMap::new(),
        title: None,
        sections: Vec::new(),
        code_blocks: Vec::new(),
        tables: Vec::new(),
        links: Vec::new(),
        images: Vec::new(),
    };

    // Extract frontmatter if present
    if config.include_metadata {
        document.metadata = extract_frontmatter(content);
    }

    let mut current_section_stack: Vec<MarkdownSection> = Vec::new();
    let mut current_text = String::new();
    let mut in_code_block = false;
    let mut current_code_block = None;
    let mut in_table = false;
    let mut current_table_headers: Vec<String> = Vec::new();
    let mut current_table_rows: Vec<Vec<String>> = Vec::new();
    let mut line_number = 1;

    for event in parser {
        match event {
            Event::Start(tag) => {
                match tag {
                    Tag::Heading(level, _, _) => {
                        let heading_level = heading_level_to_u8(level);
                        if heading_level <= config.max_heading_level {
                            // Close previous sections at same or higher level
                            close_sections_at_level(&mut current_section_stack, &mut document.sections, heading_level);
                        }
                    }
                    Tag::CodeBlock(kind) => {
                        if config.extract_code_blocks {
                            in_code_block = true;
                            let language = match kind {
                                CodeBlockKind::Fenced(lang) => {
                                    if lang.is_empty() {
                                        None
                                    } else {
                                        Some(lang.to_string())
                                    }
                                }
                                CodeBlockKind::Indented => None,
                            };

                            current_code_block = Some(CodeBlock {
                                language,
                                code: String::new(),
                                line_number,
                                id: format!("code_{}", document.code_blocks.len()),
                            });
                        }
                    }
                    Tag::Table(_) => {
                        if config.parse_tables {
                            in_table = true;
                            current_table_headers.clear();
                            current_table_rows.clear();
                        }
                    }
                    Tag::Link(_, url, title) => {
                        document.links.push(Link {
                            text: String::new(), // Will be filled in Text event
                            url: url.to_string(),
                            title: if title.is_empty() { None } else { Some(title.to_string()) },
                            is_reference: false,
                        });
                    }
                    Tag::Image(_, url, title) => {
                        document.images.push(Image {
                            alt_text: String::new(), // Will be filled in Text event
                            url: url.to_string(),
                            title: if title.is_empty() { None } else { Some(title.to_string()) },
                        });
                    }
                    _ => {}
                }
            }
            Event::End(tag) => {
                match tag {
                    Tag::Heading(level, _, _) => {
                        let heading_level = heading_level_to_u8(level);
                        if heading_level <= config.max_heading_level {
                            let section = MarkdownSection {
                                level: heading_level,
                                title: current_text.trim().to_string(),
                                content: String::new(),
                                subsections: Vec::new(),
                                id: format!("section_{}_{}", heading_level, current_section_stack.len()),
                            };

                            if document.title.is_none() && heading_level == 1 {
                                document.title = Some(section.title.clone());
                            }

                            current_section_stack.push(section);
                            current_text.clear();
                        }
                    }
                    Tag::CodeBlock(_) => {
                        if config.extract_code_blocks && in_code_block {
                            if let Some(mut code_block) = current_code_block.take() {
                                code_block.code = current_text.trim().to_string();
                                document.code_blocks.push(code_block);
                            }
                            in_code_block = false;
                            current_text.clear();
                        }
                    }
                    Tag::Table(_) => {
                        if config.parse_tables && in_table {
                            document.tables.push(MarkdownTable {
                                headers: current_table_headers.clone(),
                                rows: current_table_rows.clone(),
                                id: format!("table_{}", document.tables.len()),
                            });
                            in_table = false;
                        }
                    }
                    Tag::Paragraph => {
                        if !current_section_stack.is_empty() {
                            let last_idx = current_section_stack.len() - 1;
                            if !current_text.trim().is_empty() {
                                current_section_stack[last_idx].content.push_str(&current_text);
                                current_section_stack[last_idx].content.push('\\n');
                            }
                        }
                        current_text.clear();
                    }
                    _ => {}
                }
            }
            Event::Text(text) => {
                current_text.push_str(&text);
            }
            Event::Code(code) => {
                if config.preserve_formatting {
                    current_text.push('`');
                    current_text.push_str(&code);
                    current_text.push('`');
                } else {
                    current_text.push_str(&code);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if !in_code_block {
                    current_text.push('\\n');
                } else {
                    current_text.push('\\n');
                }
                line_number += 1;
            }
            _ => {}
        }
    }

    // Close any remaining sections
    close_sections_at_level(&mut current_section_stack, &mut document.sections, 0);

    Ok(document)
}

/// Convert markdown document to JSON file
pub fn markdown_document_to_json_file<P: AsRef<Path>>(
    document: &MarkdownDocument,
    output_file: P,
    pretty_print: bool,
) -> Result<()> {
    let file = File::create(output_file)?;
    let writer = BufWriter::new(file);

    if pretty_print {
        serde_json::to_writer_pretty(writer, document)?;
    } else {
        serde_json::to_writer(writer, document)?;
    }

    Ok(())
}

/// Extract specific sections by heading level or title pattern
pub fn extract_sections_by_pattern(
    document: &MarkdownDocument,
    pattern: &str,
    case_sensitive: bool,
) -> Vec<&MarkdownSection> {
    let mut results = Vec::new();

    fn search_sections(sections: &[MarkdownSection], pattern: &str, case_sensitive: bool, results: &mut Vec<&MarkdownSection>) {
        for section in sections {
            let title = if case_sensitive {
                &section.title
            } else {
                &section.title.to_lowercase()
            };

            let search_pattern = if case_sensitive {
                pattern
            } else {
                &pattern.to_lowercase()
            };

            if title.contains(search_pattern) {
                results.push(section);
            }

            search_sections(&section.subsections, pattern, case_sensitive, results);
        }
    }

    search_sections(&document.sections, pattern, case_sensitive, &mut results);
    results
}

/// Convert markdown tables to CSV format
pub fn tables_to_csv<P: AsRef<Path>>(
    document: &MarkdownDocument,
    output_dir: P,
) -> Result<Vec<String>> {
    use csv::Writer;

    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    let mut created_files = Vec::new();

    for (idx, table) in document.tables.iter().enumerate() {
        let filename = format!("table_{}.csv", idx);
        let filepath = output_dir.join(&filename);

        let file = File::create(&filepath)?;
        let mut writer = Writer::from_writer(BufWriter::new(file));

        // Write headers
        writer.write_record(&table.headers)?;

        // Write rows
        for row in &table.rows {
            writer.write_record(row)?;
        }

        writer.flush()?;
        created_files.push(filename);
    }

    Ok(created_files)
}

/// Extract and analyze code blocks by language
pub fn analyze_code_blocks(document: &MarkdownDocument) -> HashMap<String, Vec<&CodeBlock>> {
    let mut by_language: HashMap<String, Vec<&CodeBlock>> = HashMap::new();

    for code_block in &document.code_blocks {
        let language = code_block.language.as_deref().unwrap_or("unknown");
        by_language.entry(language.to_string()).or_insert_with(Vec::new).push(code_block);
    }

    by_language
}

/// Generate table of contents from sections
pub fn generate_toc(document: &MarkdownDocument, max_level: u8) -> String {
    let mut toc = String::new();

    fn add_sections_to_toc(sections: &[MarkdownSection], toc: &mut String, max_level: u8) {
        for section in sections {
            if section.level <= max_level {
                let indent = "  ".repeat((section.level - 1) as usize);
                toc.push_str(&format!("{}* [{}](#{})", indent, section.title, section.id));
                toc.push('\\n');

                add_sections_to_toc(&section.subsections, toc, max_level);
            }
        }
    }

    add_sections_to_toc(&document.sections, &mut toc, max_level);
    toc
}

// Helper functions

fn extract_frontmatter(content: &str) -> HashMap<String, String> {
    let mut metadata = HashMap::new();

    if content.starts_with("---") {
        if let Some(end_pos) = content[3..].find("\\n---\\n") {
            let frontmatter = &content[3..end_pos + 3];

            for line in frontmatter.lines() {
                if let Some(colon_pos) = line.find(':') {
                    let key = line[..colon_pos].trim().to_string();
                    let value = line[colon_pos + 1..].trim().trim_matches('"').to_string();
                    metadata.insert(key, value);
                }
            }
        }
    }

    metadata
}

fn heading_level_to_u8(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn close_sections_at_level(
    section_stack: &mut Vec<MarkdownSection>,
    document_sections: &mut Vec<MarkdownSection>,
    level: u8,
) {
    while let Some(section) = section_stack.last() {
        if section.level >= level && level > 0 {
            let mut completed_section = section_stack.pop().unwrap();

            if let Some(parent) = section_stack.last_mut() {
                parent.subsections.push(completed_section);
            } else {
                document_sections.push(completed_section);
            }
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_markdown() {
        let markdown = r#"# Title

This is a paragraph.

## Section 1

Content of section 1.

```rust
fn main() {
    println!("Hello, world!");
}
```

## Section 2

More content.
"#;

        let config = MarkdownConfig::default();
        let document = parse_markdown_content(markdown, &config).unwrap();

        assert_eq!(document.title, Some("Title".to_string()));
        assert_eq!(document.sections.len(), 2);
        assert_eq!(document.code_blocks.len(), 1);
        assert_eq!(document.code_blocks[0].language, Some("rust".to_string()));
    }

    #[test]
    fn test_extract_frontmatter() {
        let markdown = r#"---
title: My Document
author: John Doe
date: 2023-01-01
---

# Content

Some content here.
"#;

        let config = MarkdownConfig::default();
        let document = parse_markdown_content(markdown, &config).unwrap();

        assert_eq!(document.metadata.get("title"), Some(&"My Document".to_string()));
        assert_eq!(document.metadata.get("author"), Some(&"John Doe".to_string()));
    }

    #[test]
    fn test_nested_sections() {
        let markdown = r#"# Main Title

## Section 1

### Subsection 1.1

Content 1.1

### Subsection 1.2

Content 1.2

## Section 2

Content 2
"#;

        let config = MarkdownConfig::default();
        let document = parse_markdown_content(markdown, &config).unwrap();

        assert_eq!(document.sections.len(), 2);
        assert_eq!(document.sections[0].subsections.len(), 2);
    }

    #[test]
    fn test_generate_toc() {
        let markdown = r#"# Title

## Section 1

### Subsection 1.1

## Section 2
"#;

        let config = MarkdownConfig::default();
        let document = parse_markdown_content(markdown, &config).unwrap();
        let toc = generate_toc(&document, 3);

        assert!(toc.contains("Section 1"));
        assert!(toc.contains("Subsection 1.1"));
        assert!(toc.contains("Section 2"));
    }
}