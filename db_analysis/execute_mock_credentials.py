#!/usr/bin/env python3
"""
Execute mock credentials SQL script and verify results
"""
import sqlite3
import sys
from datetime import datetime

DB_PATH = '/app/data/settings.db'
DEFAULT_USER_ID = 'dev-user-001'

# SQL to create default user if not exists
CREATE_USER_SQL = """
INSERT OR IGNORE INTO users (
    user_id,
    pubkey,
    tier,
    email,
    username,
    display_name,
    is_active,
    is_verified,
    rate_limit_tier
) VALUES (
    'dev-user-001',
    'mock_pubkey_for_development',
    'public',
    'dev@localhost',
    'developer',
    'Development User',
    1,
    1,
    'free'
);
"""

# SQL INSERT statements
INSERT_SQL = """
INSERT INTO api_keys (
    user_id,
    service_name,
    api_key_encrypted,
    key_hash,
    key_name,
    key_description,
    scopes,
    is_active,
    created_at,
    updated_at
) VALUES
-- Nostr Relay (Decentralized Social Protocol)
(
    'dev-user-001',
    'nostr',
    'wss://relay.damus.io',
    'mock_hash_nostr',
    'Mock Nostr Relay',
    'Development relay for testing decentralized features',
    'read,write,publish',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
),

-- GitHub Integration
(
    'dev-user-001',
    'github',
    'ghp_mock_development_token_DO_NOT_USE_IN_PRODUCTION',
    'mock_hash_github',
    'Mock GitHub Token',
    'Development token for repository integration testing',
    'repo,read:org,read:user',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
),

-- RAGFlow (Retrieval-Augmented Generation)
(
    'dev-user-001',
    'ragflow',
    'mock_ragflow_api_key_development_only',
    'mock_hash_ragflow',
    'Mock RAGFlow API',
    'Development API key for RAG system testing',
    'read,write,query',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
),

-- Anthropic Claude API
(
    'dev-user-001',
    'anthropic',
    'sk-ant-mock-development-key-DO_NOT_USE',
    'mock_hash_anthropic',
    'Mock Claude API',
    'Development API key for Claude integration testing',
    'messages,completions',
    1,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);
"""

def main():
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        print("=" * 60)
        print("VisionFlow Mock Credentials Setup")
        print("=" * 60)
        print()

        # Check if is_mock column exists (it may not based on schema)
        cursor.execute("PRAGMA table_info(api_keys)")
        columns = [col[1] for col in cursor.fetchall()]
        has_is_mock = 'is_mock' in columns

        # Create default user first
        print("Creating default user...")
        cursor.executescript(CREATE_USER_SQL)
        conn.commit()
        print("✓ Default user created/verified")
        print()

        # Execute INSERT
        print("Inserting mock credentials...")
        cursor.executescript(INSERT_SQL)
        conn.commit()
        print("✓ Mock credentials inserted successfully")
        print()

        # Verify insertion
        print("Verifying credentials...")
        print("-" * 60)

        if has_is_mock:
            cursor.execute("""
                SELECT service_name, key_name, is_mock, is_active
                FROM api_keys
                ORDER BY created_at DESC
            """)
            print(f"{'Service':<15} {'Key Name':<30} {'Mock':<8} {'Active'}")
            print("-" * 60)
            for row in cursor.fetchall():
                service, key_name, is_mock, is_active = row
                mock_status = "Yes" if is_mock else "No"
                active_status = "✓" if is_active else "✗"
                print(f"{service:<15} {key_name:<30} {mock_status:<8} {active_status}")
        else:
            cursor.execute("""
                SELECT service_name, key_name, is_active
                FROM api_keys
                ORDER BY created_at DESC
            """)
            print(f"{'Service':<15} {'Key Name':<30} {'Active'}")
            print("-" * 60)
            for row in cursor.fetchall():
                service, key_name, is_active = row
                active_status = "✓" if is_active else "✗"
                print(f"{service:<15} {key_name:<30} {active_status}")

        print()

        # Count total
        cursor.execute("SELECT COUNT(*) FROM api_keys")
        total = cursor.fetchone()[0]
        print(f"Total API keys in database: {total}")
        print()

        # Show mock credentials details
        print("=" * 60)
        print("Mock Credentials Details")
        print("=" * 60)
        cursor.execute("""
            SELECT service_name, key_name, key_description, scopes
            FROM api_keys
            ORDER BY created_at DESC
            LIMIT 4
        """)

        for service, key_name, description, scopes in cursor.fetchall():
            print(f"\nService: {service}")
            print(f"  Name: {key_name}")
            print(f"  Description: {description}")
            print(f"  Scopes: {scopes}")

        conn.close()

        print()
        print("=" * 60)
        print("✓ Mock credentials setup complete!")
        print("=" * 60)

        return 0

    except sqlite3.IntegrityError as e:
        print(f"✗ Error: {e}")
        print("Credentials may already exist in the database.")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
