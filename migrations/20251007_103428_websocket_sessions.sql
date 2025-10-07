-- WebSocket session tracking
CREATE TABLE IF NOT EXISTS websocket_sessions (
    id TEXT PRIMARY KEY,
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_ping TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subscribed_to_telemetry BOOLEAN DEFAULT FALSE
);

-- Agent to session mapping
CREATE TABLE IF NOT EXISTS agent_session_subscriptions (
    agent_id INTEGER,
    session_id TEXT,
    subscribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, session_id),
    FOREIGN KEY (agent_id) REFERENCES bots_agents(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES websocket_sessions(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ws_sessions_active ON websocket_sessions(connected_at, last_ping);
CREATE INDEX IF NOT EXISTS idx_agent_subs_session ON agent_session_subscriptions(session_id);
