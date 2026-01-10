pub mod batch;
pub mod core;
pub mod distill;
pub mod embed;
pub mod llm;
pub mod queue;
pub mod session;
pub mod store;
pub mod transcript;
pub mod watch;

pub use batch::{BatchError, BatchQueue, BatchRequest, BatchResultItem, PendingBatch};
pub use core::{Node, SourceType, EMBEDDING_DIM};
pub use distill::{
    extract_knowledge, DistillMode, ExtractedInsight, ExtractionResult,
    CHUNK_THRESHOLD_TOKENS, EXTRACTION_SYSTEM_PROMPT,
};
pub use llm::{LlmError, LlmResult};
pub use embed::{chunk_text, embed, EmbedError};
pub use queue::{Job, JobStatus, Queue};
pub use session::{discover_sessions, discover_sessions_in_dir, list_all_projects, SessionInfo};
pub use store::{Processed, Store};
pub use transcript::{format_context, format_context_with_options, FormatOptions, read_transcript, TranscriptEntry};
pub use watch::{AllProjectsWatcher, ProjectSessionEvent, SessionEvent, SessionWatcher};
