use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Target size for each chunk in characters
    pub target_size: usize,
    /// Overlap between consecutive chunks in characters
    pub overlap: usize,
    /// Minimum size for a chunk (prevents very small chunks)
    pub min_size: usize,
    /// Include heading context as prefix in chunk text
    pub include_heading_context: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 1000,
            overlap: 200,
            min_size: 100,
            include_heading_context: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    /// The text content of the chunk
    pub text: String,
    /// Heading hierarchy path (e.g., ["Introduction", "Background"])
    pub heading_path: Vec<String>,
    /// Start offset in the original document
    pub start_offset: usize,
    /// End offset in the original document
    pub end_offset: usize,
    /// Index of this chunk in the document
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Frontmatter {
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub date: Option<String>,
    pub custom: HashMap<String, String>,
}

#[derive(Debug, Error)]
pub enum ChunkError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid frontmatter: {0}")]
    InvalidFrontmatter(String),
}

pub struct MarkdownChunker {
    config: ChunkConfig,
}

impl MarkdownChunker {
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Parse frontmatter from the beginning of markdown content
    pub(crate) fn parse_frontmatter(content: &str) -> (Option<Frontmatter>, &str) {
        if !content.starts_with("---") {
            return (None, content);
        }

        let rest = &content[3..];
        if let Some(end_pos) = rest.find("\n---") {
            let frontmatter_str = &rest[..end_pos];
            let remaining = &rest[end_pos + 4..];

            let mut frontmatter = Frontmatter::default();

            for line in frontmatter_str.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if let Some((key, value)) = line.split_once(':') {
                    let key = key.trim();
                    let value = value.trim();

                    match key {
                        "title" => frontmatter.title = Some(value.trim_matches('"').to_string()),
                        "date" => frontmatter.date = Some(value.to_string()),
                        "tags" => {
                            let tags = value.trim_matches(|c| c == '[' || c == ']');
                            frontmatter.tags = tags
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect();
                        }
                        _ => {
                            frontmatter.custom.insert(key.to_string(), value.to_string());
                        }
                    }
                }
            }

            (Some(frontmatter), remaining)
        } else {
            (None, content)
        }
    }

    /// Split markdown into semantic chunks
    pub fn chunk(&self, markdown: &str) -> Vec<SemanticChunk> {
        let (_frontmatter, content) = Self::parse_frontmatter(markdown);

        let parser = Parser::new(content);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut heading_stack: Vec<String> = Vec::new();
        let mut current_offset = 0;
        let mut chunk_start_offset = 0;
        let mut in_code_block = false;
        let mut code_block_content = String::new();

        for event in parser {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    // Finalize current chunk before starting new section
                    if !current_chunk.is_empty() && current_chunk.len() >= self.config.min_size {
                        chunks.push(SemanticChunk {
                            text: current_chunk.clone(),
                            heading_path: heading_stack.clone(),
                            start_offset: chunk_start_offset,
                            end_offset: current_offset,
                            chunk_index: chunks.len(),
                        });
                        current_chunk.clear();
                        chunk_start_offset = current_offset;
                    }

                    // Adjust heading stack
                    let level_idx = level as usize;
                    heading_stack.truncate(level_idx.saturating_sub(1));
                }
                Event::End(TagEnd::Heading(level)) => {
                    let heading_text = current_chunk.trim().to_string();
                    let level_idx = level as usize;

                    if level_idx > 0 {
                        if heading_stack.len() >= level_idx {
                            heading_stack[level_idx - 1] = heading_text.clone();
                        } else {
                            heading_stack.push(heading_text.clone());
                        }
                    }

                    if self.config.include_heading_context && !heading_text.is_empty() {
                        current_chunk = format!("{}\n\n", heading_text);
                    } else {
                        current_chunk.clear();
                    }
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    in_code_block = true;
                    code_block_content.clear();
                }
                Event::End(TagEnd::CodeBlock) => {
                    in_code_block = false;
                    // Treat code blocks as atomic units
                    current_chunk.push_str(&format!("```\n{}\n```\n", code_block_content));
                    code_block_content.clear();
                }
                Event::Text(text) => {
                    if in_code_block {
                        code_block_content.push_str(&text);
                    } else {
                        current_chunk.push_str(&text);
                    }
                    current_offset += text.len();
                }
                Event::Code(code) => {
                    current_chunk.push('`');
                    current_chunk.push_str(&code);
                    current_chunk.push('`');
                    current_offset += code.len() + 2;
                }
                Event::SoftBreak | Event::HardBreak => {
                    current_chunk.push('\n');
                    current_offset += 1;
                }
                Event::Start(Tag::Paragraph) => {
                    if !current_chunk.is_empty() && !current_chunk.ends_with('\n') {
                        current_chunk.push('\n');
                    }
                }
                Event::End(TagEnd::Paragraph) => {
                    current_chunk.push_str("\n\n");
                    current_offset += 2;

                    // Check if we should split here
                    if current_chunk.len() >= self.config.target_size {
                        chunks.push(SemanticChunk {
                            text: current_chunk.clone(),
                            heading_path: heading_stack.clone(),
                            start_offset: chunk_start_offset,
                            end_offset: current_offset,
                            chunk_index: chunks.len(),
                        });

                        // Apply overlap
                        if self.config.overlap > 0 {
                            let overlap_text = current_chunk
                                .chars()
                                .rev()
                                .take(self.config.overlap)
                                .collect::<String>()
                                .chars()
                                .rev()
                                .collect::<String>();
                            current_chunk = overlap_text;
                        } else {
                            current_chunk.clear();
                        }
                        chunk_start_offset = current_offset - current_chunk.len();
                    }
                }
                _ => {}
            }
        }

        // Add final chunk
        if !current_chunk.is_empty() && current_chunk.len() >= self.config.min_size {
            chunks.push(SemanticChunk {
                text: current_chunk,
                heading_path: heading_stack,
                start_offset: chunk_start_offset,
                end_offset: current_offset,
                chunk_index: chunks.len(),
            });
        }

        chunks
    }

    /// Read and chunk a markdown file
    pub fn chunk_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(Option<Frontmatter>, Vec<SemanticChunk>), ChunkError> {
        let content = std::fs::read_to_string(path)?;
        let (frontmatter, remaining) = Self::parse_frontmatter(&content);
        let chunks = self.chunk(remaining);
        Ok((frontmatter, chunks))
    }
}

