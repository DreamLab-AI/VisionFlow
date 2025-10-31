-- ==============================================================================
-- CONTROL CENTER SCHEMA: Physics, Constraint, and Rendering Settings
-- ==============================================================================
-- Purpose: User-tunable parameters for GPU physics and visualization
-- Integration: Part of unified.db but documented separately for clarity
-- Date: 2025-10-31
-- ==============================================================================

-- NOTE: This file is extracted from unified_schema.sql for architectural clarity.
-- In production, these tables are part of unified.db.

-- ==============================================================================
-- 1. PHYSICS SETTINGS: GPU force computation parameters
-- ==============================================================================

CREATE TABLE physics_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,           -- e.g., "default", "high-performance", "precision"

    -- Time Stepping
    dt REAL DEFAULT 0.2,                         -- Time step (seconds per iteration)
    max_iterations INTEGER DEFAULT 300,          -- Maximum physics iterations per session

    -- Force Parameters
    spring_k REAL DEFAULT 0.01,                  -- Spring stiffness (edge attraction)
    repel_k REAL DEFAULT 500.0,                  -- Coulomb repulsion strength
    center_gravity_k REAL DEFAULT 0.001,         -- Centering force (prevent drift)

    -- Damping (Energy Dissipation)
    damping REAL DEFAULT 0.85,                   -- Global velocity decay (0.85 = 15% loss/frame)
    warmup_damping REAL DEFAULT 0.5,             -- Extra damping during first N iterations
    warmup_iterations INTEGER DEFAULT 50,        -- Warmup period length
    cooling_rate REAL DEFAULT 0.0,               -- Adaptive cooling (0.0 = disabled)

    -- Force Limits (Stability)
    max_velocity REAL DEFAULT 50.0,              -- Velocity clamp (units/frame)
    max_force REAL DEFAULT 15.0,                 -- Force clamp (acceleration units)

    -- Boundary Handling
    boundary_limit REAL DEFAULT 5000.0,          -- Soft boundary radius
    boundary_damping REAL DEFAULT 0.3,           -- Boundary repulsion strength

    -- Spatial Grid (for O(n) collision detection)
    grid_cell_size REAL DEFAULT 100.0,           -- Uniform grid cell size (auto-tuned if NULL)

    -- Mass Scaling
    mass_scale REAL DEFAULT 1.0,                 -- Global mass multiplier (for graph-wide tuning)
    charge_scale REAL DEFAULT 1.0,               -- Global charge multiplier

    -- Stability Gates (auto-pause physics when stable)
    enable_stability_gates BOOLEAN DEFAULT 1,
    stability_threshold REAL DEFAULT 0.001,      -- Avg kinetic energy threshold
    stability_min_iterations INTEGER DEFAULT 600,-- Min iterations before early exit

    -- SSSP Integration
    enable_sssp_springs BOOLEAN DEFAULT 0,       -- Use graph distance for rest lengths
    sssp_multiplier REAL DEFAULT 10.0,           -- distance = graph_dist * multiplier

    -- Advanced
    enable_barnes_hut BOOLEAN DEFAULT 0,         -- Use octree for O(n log n) repulsion
    barnes_hut_theta REAL DEFAULT 0.5,           -- Theta parameter (accuracy vs speed)

    -- Metadata
    description TEXT,
    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_physics_profile ON physics_settings(profile_name);
CREATE INDEX idx_physics_default ON physics_settings(is_default);

-- ==============================================================================
-- 2. CONSTRAINT SETTINGS: Ontology → Physics translation parameters
-- ==============================================================================

CREATE TABLE constraint_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,

    -- Progressive Activation (smooth constraint ramping)
    enable_progressive_activation BOOLEAN DEFAULT 1,
    constraint_ramp_frames INTEGER DEFAULT 60,   -- Frames to reach full strength

    -- LOD (Level of Detail) Thresholds
    -- Constraints are culled based on camera distance to reduce computation
    lod_near_distance REAL DEFAULT 100.0,        -- Distance < 100: All constraints active
    lod_medium_distance REAL DEFAULT 500.0,      -- Distance 100-500: Priority ≤ 5 active
    lod_far_distance REAL DEFAULT 1000.0,        -- Distance 500-1000: Priority ≤ 3 active
    lod_culling_distance REAL DEFAULT 2000.0,    -- Distance > 2000: Only priority 1 active

    -- Priority Weights (for conflict resolution)
    -- Weight = 10^(-(priority - 1) / 9)
    priority_1_weight REAL DEFAULT 10.0,         -- User overrides (highest)
    priority_2_weight REAL DEFAULT 5.0,          -- Identity constraints (SameAs)
    priority_3_weight REAL DEFAULT 2.5,          -- Disjoint constraints
    priority_4_weight REAL DEFAULT 1.5,          -- Hierarchy constraints (SubClassOf)
    priority_5_weight REAL DEFAULT 1.0,          -- Default constraints
    priority_6_weight REAL DEFAULT 0.7,          -- Inferred constraints
    priority_7_weight REAL DEFAULT 0.5,          -- Soft alignment
    priority_8_weight REAL DEFAULT 0.3,          -- Base physics
    priority_9_weight REAL DEFAULT 0.1,          -- Optimization hints
    priority_10_weight REAL DEFAULT 0.05,        -- Background tasks

    -- Constraint Type Parameters
    -- DisjointClasses
    disjoint_min_distance REAL DEFAULT 80.0,     -- Minimum separation
    disjoint_strength REAL DEFAULT 0.9,

    -- SubClassOf
    subclass_ideal_distance REAL DEFAULT 30.0,   -- Parent-child spacing
    subclass_strength REAL DEFAULT 0.7,

    -- SameIndividual
    same_target_distance REAL DEFAULT 2.0,       -- Co-location tolerance
    same_strength REAL DEFAULT 1.0,

    -- FunctionalProperty
    functional_boundary_size REAL DEFAULT 20.0,  -- Containment radius
    functional_strength REAL DEFAULT 0.8,

    -- Activation Frame Offset
    activation_frame_offset INTEGER DEFAULT 0,   -- Global frame to start constraints

    -- Conflict Resolution
    enable_weighted_blending BOOLEAN DEFAULT 1,  -- Blend conflicting constraints
    conflict_resolution_method TEXT DEFAULT 'weighted', -- 'weighted', 'priority', 'user_only'

    -- Telemetry
    enable_violation_tracking BOOLEAN DEFAULT 1, -- Track constraint satisfaction
    enable_energy_tracking BOOLEAN DEFAULT 1,    -- Track constraint energy

    -- Metadata
    description TEXT,
    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_constraint_profile ON constraint_settings(profile_name);
CREATE INDEX idx_constraint_default ON constraint_settings(is_default);

-- ==============================================================================
-- 3. RENDERING SETTINGS: Visualization LOD and quality
-- ==============================================================================

CREATE TABLE rendering_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,

    -- Node Rendering
    node_size_scale REAL DEFAULT 1.0,            -- Global node size multiplier
    node_base_size REAL DEFAULT 10.0,            -- Base node size (before scaling)

    -- Node LOD (Level of Detail) based on camera distance
    node_lod_near REAL DEFAULT 10.0,             -- Full detail (all geometry)
    node_lod_medium REAL DEFAULT 50.0,           -- Medium detail (simplified mesh)
    node_lod_far REAL DEFAULT 200.0,             -- Low detail (billboard/sphere)
    node_lod_cull REAL DEFAULT 1000.0,           -- Culling distance (not rendered)

    -- Edge Rendering
    edge_thickness REAL DEFAULT 1.0,             -- Line thickness
    edge_lod_near REAL DEFAULT 50.0,             -- All edges visible
    edge_lod_far REAL DEFAULT 500.0,             -- Only important edges
    edge_lod_cull REAL DEFAULT 2000.0,           -- No edges rendered

    -- Label Rendering
    label_distance_min REAL DEFAULT 10.0,        -- Always visible if camera closer than this
    label_distance_max REAL DEFAULT 100.0,       -- Always hidden if camera farther than this
    label_importance_threshold REAL DEFAULT 0.5, -- Semantic importance cutoff (0-1)
    label_max_visible INTEGER DEFAULT 100,       -- Max labels shown at once

    -- Colors
    default_node_color TEXT DEFAULT '#3498db',   -- Blue
    default_edge_color TEXT DEFAULT '#95a5a6',   -- Gray
    highlight_color TEXT DEFAULT '#e74c3c',      -- Red
    pinned_node_color TEXT DEFAULT '#f39c12',    -- Orange
    selected_node_color TEXT DEFAULT '#2ecc71',  -- Green

    -- Opacity
    default_node_opacity REAL DEFAULT 1.0,
    default_edge_opacity REAL DEFAULT 0.5,
    pinned_node_opacity REAL DEFAULT 1.0,

    -- Performance Limits
    max_visible_nodes INTEGER DEFAULT 5000,      -- Cull beyond this count
    max_visible_edges INTEGER DEFAULT 10000,
    max_draw_calls INTEGER DEFAULT 1000,         -- Babylon.js limit

    -- Frustum Culling
    enable_frustum_culling BOOLEAN DEFAULT 1,
    frustum_padding REAL DEFAULT 1.2,            -- Padding multiplier (1.2 = 20% extra)

    -- Shadows and Post-Processing
    enable_shadows BOOLEAN DEFAULT 0,            -- Expensive
    enable_bloom BOOLEAN DEFAULT 1,
    enable_antialiasing BOOLEAN DEFAULT 1,

    -- Metadata
    description TEXT,
    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rendering_profile ON rendering_settings(profile_name);
CREATE INDEX idx_rendering_default ON rendering_settings(is_default);

-- ==============================================================================
-- 4. USER PROFILES: Save/load complete configurations
-- ==============================================================================

CREATE TABLE constraint_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,                                -- User identifier (optional, for multi-user)
    profile_name TEXT NOT NULL,

    -- Profile References (FK to settings tables)
    physics_profile TEXT,
    constraint_profile TEXT,
    rendering_profile TEXT,

    -- Metadata
    description TEXT,
    is_public BOOLEAN DEFAULT 0,                 -- Share with other users?

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (physics_profile) REFERENCES physics_settings(profile_name) ON DELETE SET NULL,
    FOREIGN KEY (constraint_profile) REFERENCES constraint_settings(profile_name) ON DELETE SET NULL,
    FOREIGN KEY (rendering_profile) REFERENCES rendering_settings(profile_name) ON DELETE SET NULL
);

CREATE INDEX idx_profiles_user ON constraint_profiles(user_id);
CREATE INDEX idx_profiles_name ON constraint_profiles(profile_name);
CREATE INDEX idx_profiles_public ON constraint_profiles(is_public);

-- ==============================================================================
-- 5. DEFAULT DATA: Production-tuned presets
-- ==============================================================================

-- Default Physics Profile (empirically tuned for stable layouts)
INSERT INTO physics_settings (
    profile_name, dt, spring_k, repel_k, center_gravity_k,
    damping, warmup_damping, warmup_iterations,
    max_velocity, max_force,
    boundary_limit, boundary_damping,
    grid_cell_size,
    enable_stability_gates, stability_threshold, stability_min_iterations,
    description, is_default
) VALUES (
    'default',
    0.2,          -- dt
    0.01,         -- spring_k
    500.0,        -- repel_k
    0.001,        -- center_gravity_k
    0.85,         -- damping
    0.5,          -- warmup_damping
    50,           -- warmup_iterations
    50.0,         -- max_velocity
    15.0,         -- max_force
    5000.0,       -- boundary_limit
    0.3,          -- boundary_damping
    100.0,        -- grid_cell_size
    1,            -- enable_stability_gates
    0.001,        -- stability_threshold
    600,          -- stability_min_iterations
    'Default physics profile - balanced performance and quality',
    1             -- is_default
);

-- High-Performance Profile (speed over precision)
INSERT INTO physics_settings (
    profile_name, dt, spring_k, repel_k, damping,
    max_velocity, max_force, grid_cell_size,
    enable_stability_gates, stability_threshold,
    description, is_default
) VALUES (
    'high-performance',
    0.3,          -- Larger timestep
    0.02,         -- Stiffer springs (faster convergence)
    400.0,        -- Lower repulsion
    0.9,          -- Higher damping
    100.0,        -- Higher velocity
    20.0,         -- Higher force
    150.0,        -- Larger grid cells
    1,            -- enable_stability_gates
    0.01,         -- Coarser stability threshold
    'High-performance profile - prioritizes FPS over layout quality',
    0
);

-- Precision Profile (quality over speed)
INSERT INTO physics_settings (
    profile_name, dt, spring_k, repel_k, damping,
    max_velocity, max_force, grid_cell_size,
    enable_stability_gates, stability_threshold,
    description, is_default
) VALUES (
    'precision',
    0.1,          -- Smaller timestep
    0.005,        -- Softer springs
    600.0,        -- Higher repulsion
    0.8,          -- Lower damping (more movement)
    30.0,         -- Lower velocity
    10.0,         -- Lower force
    75.0,         -- Smaller grid cells
    1,            -- enable_stability_gates
    0.0001,       -- Fine-grained stability threshold
    'Precision profile - prioritizes layout quality over speed',
    0
);

-- Default Constraint Profile
INSERT INTO constraint_settings (
    profile_name,
    enable_progressive_activation, constraint_ramp_frames,
    lod_near_distance, lod_medium_distance, lod_far_distance,
    disjoint_min_distance, disjoint_strength,
    subclass_ideal_distance, subclass_strength,
    same_target_distance, same_strength,
    description, is_default
) VALUES (
    'default',
    1,            -- enable_progressive_activation
    60,           -- constraint_ramp_frames
    100.0,        -- lod_near_distance
    500.0,        -- lod_medium_distance
    1000.0,       -- lod_far_distance
    80.0,         -- disjoint_min_distance
    0.9,          -- disjoint_strength
    30.0,         -- subclass_ideal_distance
    0.7,          -- subclass_strength
    2.0,          -- same_target_distance
    1.0,          -- same_strength
    'Default constraint profile - balanced semantic fidelity',
    1             -- is_default
);

-- Default Rendering Profile
INSERT INTO rendering_settings (
    profile_name,
    node_size_scale, node_base_size,
    node_lod_near, node_lod_medium, node_lod_far, node_lod_cull,
    edge_thickness,
    edge_lod_near, edge_lod_far,
    label_distance_min, label_distance_max,
    max_visible_nodes, max_visible_edges,
    enable_frustum_culling,
    description, is_default
) VALUES (
    'default',
    1.0,          -- node_size_scale
    10.0,         -- node_base_size
    10.0,         -- node_lod_near
    50.0,         -- node_lod_medium
    200.0,        -- node_lod_far
    1000.0,       -- node_lod_cull
    1.0,          -- edge_thickness
    50.0,         -- edge_lod_near
    500.0,        -- edge_lod_far
    10.0,         -- label_distance_min
    100.0,        -- label_distance_max
    5000,         -- max_visible_nodes
    10000,        -- max_visible_edges
    1,            -- enable_frustum_culling
    'Default rendering profile - balanced quality and performance',
    1             -- is_default
);

-- ==============================================================================
-- 6. TRIGGERS: Auto-update timestamps
-- ==============================================================================

CREATE TRIGGER update_physics_timestamp
AFTER UPDATE ON physics_settings
FOR EACH ROW
BEGIN
    UPDATE physics_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_constraint_timestamp
AFTER UPDATE ON constraint_settings
FOR EACH ROW
BEGIN
    UPDATE constraint_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_rendering_timestamp
AFTER UPDATE ON rendering_settings
FOR EACH ROW
BEGIN
    UPDATE rendering_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_profile_timestamp
AFTER UPDATE ON constraint_profiles
FOR EACH ROW
BEGIN
    UPDATE constraint_profiles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ==============================================================================
-- 7. VIEWS: Convenient access patterns
-- ==============================================================================

-- Complete profile view (join all settings)
CREATE VIEW complete_profile_view AS
SELECT
    p.id AS profile_id,
    p.user_id,
    p.profile_name,
    p.description AS profile_description,

    -- Physics settings
    ph.dt, ph.spring_k, ph.repel_k, ph.damping,
    ph.max_velocity, ph.max_force,
    ph.enable_stability_gates,

    -- Constraint settings
    cs.enable_progressive_activation,
    cs.constraint_ramp_frames,
    cs.lod_near_distance,

    -- Rendering settings
    rs.node_size_scale,
    rs.edge_thickness,
    rs.max_visible_nodes

FROM constraint_profiles p
LEFT JOIN physics_settings ph ON p.physics_profile = ph.profile_name
LEFT JOIN constraint_settings cs ON p.constraint_profile = cs.profile_name
LEFT JOIN rendering_settings rs ON p.rendering_profile = rs.profile_name;

-- Active settings view (default profiles)
CREATE VIEW active_settings AS
SELECT
    'physics' AS setting_type,
    profile_name,
    description,
    created_at
FROM physics_settings
WHERE is_default = 1

UNION ALL

SELECT
    'constraint' AS setting_type,
    profile_name,
    description,
    created_at
FROM constraint_settings
WHERE is_default = 1

UNION ALL

SELECT
    'rendering' AS setting_type,
    profile_name,
    description,
    created_at
FROM rendering_settings
WHERE is_default = 1;

-- ==============================================================================
-- 8. API EXAMPLES: How to use these tables
-- ==============================================================================

/*
-- Get default physics parameters
SELECT * FROM physics_settings WHERE is_default = 1;

-- Create a new user profile
INSERT INTO constraint_profiles (user_id, profile_name, physics_profile, constraint_profile, rendering_profile, description)
VALUES ('user123', 'My Custom Layout', 'high-performance', 'default', 'default', 'Fast layout for large graphs');

-- Update a specific physics parameter
UPDATE physics_settings
SET repel_k = 600.0
WHERE profile_name = 'default';

-- Get constraint LOD thresholds for distance-based culling
SELECT lod_near_distance, lod_medium_distance, lod_far_distance
FROM constraint_settings
WHERE profile_name = 'default';

-- Switch to precision profile
UPDATE physics_settings SET is_default = 0;
UPDATE physics_settings SET is_default = 1 WHERE profile_name = 'precision';
*/

-- ==============================================================================
-- END OF CONTROL CENTER SCHEMA
-- ==============================================================================
