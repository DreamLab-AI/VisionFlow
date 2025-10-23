-- VisionFlow Mock Credentials Setup Script
-- ==========================================
-- USE FOR DEVELOPMENT/TESTING ONLY
-- Run inside container: docker exec -it visionflow_container sqlite3 /app/data/settings.db < add_mock_credentials.sql

-- Insert mock API credentials
INSERT INTO api_keys (
    service_name,
    api_key_encrypted,
    key_name,
    key_description,
    scopes,
    is_active,
    created_at,
    updated_at
) VALUES
-- Nostr Relay (Decentralized Social Protocol)
(
    'nostr',
    'wss://relay.damus.io',
    'Mock Nostr Relay',
    'Development relay for testing decentralized features',
    'read,write,publish',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
),

-- GitHub Integration
(
    'github',
    'ghp_mock_development_token_DO_NOT_USE_IN_PRODUCTION',
    'Mock GitHub Token',
    'Development token for repository integration testing',
    'repo,read:org,read:user',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
),

-- RAGFlow (Retrieval-Augmented Generation)
(
    'ragflow',
    'mock_ragflow_api_key_development_only',
    'Mock RAGFlow API',
    'Development API key for RAG system testing',
    'read,write,query',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
),

-- Anthropic Claude API
(
    'anthropic',
    'sk-ant-mock-development-key-DO_NOT_USE',
    'Mock Claude API',
    'Development API key for Claude integration testing',
    'messages,completions',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- Verify insertion
SELECT
    service_name,
    key_name,
    scopes,
    is_active,
    created_at
FROM api_keys
ORDER BY created_at DESC;

-- Show count
SELECT COUNT(*) as total_api_keys FROM api_keys;
