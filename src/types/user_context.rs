//! User context for cross-boundary identity propagation.
//!
//! When a VisionFlow user triggers agent work (via UI, voice, or API), this
//! struct carries their identity across the container boundary into the
//! multi-agent Docker environment. Agents use this to create user-scoped
//! workspaces, attribute beads, and file briefing responses.

use serde::{Deserialize, Serialize};

/// User identity context propagated to the agent container.
///
/// Built from `NostrUser` in the Rust backend and included in every
/// Management API task creation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// Bech32-encoded Nostr public key (npub1...). Primary user identifier.
    pub user_id: String,

    /// Hex-encoded secp256k1 public key. Used for cryptographic verification.
    pub pubkey: String,

    /// Human-readable display name, used for folder paths (e.g., "dinis_cruz").
    /// Derived from Nostr profile metadata or configured per user.
    pub display_name: String,

    /// Active session UUID. Ties the agent task back to the originating session.
    pub session_id: String,

    /// Whether this user has elevated privileges (from POWER_USER_PUBKEYS).
    pub is_power_user: bool,
}

/// Options for task creation that extend the base agent/task/provider triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTaskOptions {
    /// The agent skill to invoke (e.g., "ontology-core", "build-with-quality").
    pub agent: String,

    /// The task description or prompt for the agent.
    pub task: String,

    /// The execution provider (e.g., "claude-flow", "gemini").
    pub provider: String,

    /// User context for identity propagation. None for system-initiated tasks.
    pub user_context: Option<UserContext>,

    /// Whether to create a Beads epic for this task.
    pub with_beads: bool,

    /// Optional brief ID if this task was triggered by a briefing workflow.
    pub brief_id: Option<String>,

    /// Optional parent bead ID for sub-task creation (swarm/team coordination).
    pub parent_bead_id: Option<String>,
}

/// Briefing request from a user — the input to the briefing workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefingRequest {
    /// The brief content (markdown). May come from voice transcription + refinement.
    pub content: String,

    /// Roles to involve in the response (e.g., ["architect", "dev", "ciso"]).
    pub roles: Vec<String>,

    /// Optional version string (e.g., "v0.2.33").
    pub version: Option<String>,

    /// Optional brief type (e.g., "daily-brief", "feature-request").
    pub brief_type: Option<String>,

    /// Optional slug for the filename.
    pub slug: Option<String>,
}

/// Response from brief execution — links to the debrief and individual role responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefingResponse {
    /// The brief ID (for tracking).
    pub brief_id: String,

    /// Path to the brief file in the repo.
    pub brief_path: String,

    /// Bead ID for the epic tracking this brief.
    pub bead_id: Option<String>,

    /// Task IDs spawned per role.
    pub role_tasks: Vec<RoleTask>,
}

/// A single role's task within a briefing workflow execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleTask {
    /// The role name (e.g., "architect").
    pub role: String,

    /// The Management API task ID.
    pub task_id: String,

    /// The bead ID for this role's sub-task.
    pub bead_id: Option<String>,

    /// Path where the role will write its response.
    pub response_path: String,
}
