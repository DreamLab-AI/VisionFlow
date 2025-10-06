-- Initialize Vircadia World Server Database
-- This script runs on first PostgreSQL startup

-- Create Vircadia database (if not exists)
SELECT 'CREATE DATABASE vircadia_world'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'vircadia_world')\gexec

-- Connect to vircadia_world database
\c vircadia_world;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Worlds table
CREATE TABLE IF NOT EXISTS worlds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id VARCHAR(255),
    max_users INTEGER DEFAULT 50,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Entities table (3D objects in world)
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID REFERENCES worlds(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL,
    name VARCHAR(255),
    position_x DOUBLE PRECISION DEFAULT 0,
    position_y DOUBLE PRECISION DEFAULT 0,
    position_z DOUBLE PRECISION DEFAULT 0,
    rotation_x DOUBLE PRECISION DEFAULT 0,
    rotation_y DOUBLE PRECISION DEFAULT 0,
    rotation_z DOUBLE PRECISION DEFAULT 0,
    rotation_w DOUBLE PRECISION DEFAULT 1,
    scale_x DOUBLE PRECISION DEFAULT 1,
    scale_y DOUBLE PRECISION DEFAULT 1,
    scale_z DOUBLE PRECISION DEFAULT 1,
    metadata JSONB,
    owner_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Spatial index for entities
CREATE INDEX IF NOT EXISTS entities_spatial_idx ON entities USING gist (
    cube(
        ARRAY[position_x, position_y, position_z],
        ARRAY[position_x, position_y, position_z]
    )
);

-- Sessions table (user connections)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID REFERENCES worlds(id) ON DELETE CASCADE,
    agent_id VARCHAR(255) NOT NULL,
    username VARCHAR(255),
    connected_at TIMESTAMP DEFAULT NOW(),
    last_seen_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Annotations table (collaborative notes)
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID REFERENCES worlds(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    agent_id VARCHAR(255) NOT NULL,
    username VARCHAR(255),
    text TEXT NOT NULL,
    position_x DOUBLE PRECISION,
    position_y DOUBLE PRECISION,
    position_z DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert default world
INSERT INTO worlds (id, name, description, owner_id)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'VisionFlow World',
    'Default multi-user world for VisionFlow agent swarm visualization',
    'system'
) ON CONFLICT DO NOTHING;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_entities_world_id ON entities(world_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_sessions_world_id ON sessions(world_id);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_annotations_world_id ON annotations(world_id);
CREATE INDEX IF NOT EXISTS idx_annotations_entity_id ON annotations(entity_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE vircadia_world TO visionflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO visionflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO visionflow;
