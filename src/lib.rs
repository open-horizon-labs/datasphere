pub mod core;
pub mod distill;
pub mod embed;
pub mod llm;
pub mod session;
pub mod store;
pub mod transcript;
pub mod watch;

pub use core::{Edge, Node, SourceType, EMBEDDING_DIM};
pub use distill::{extract_knowledge, ExtractedInsight, ExtractionResult};
pub use embed::{embed, EmbedError};
pub use session::{discover_sessions, SessionInfo};
pub use store::{Processed, Store};
pub use transcript::{format_context, read_transcript, TranscriptEntry};
pub use watch::{SessionEvent, SessionWatcher};
