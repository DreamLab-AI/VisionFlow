#!/usr/bin/env python3
"""
Agentic Lightning MCP Server

RL training integration leveraging AgentDB + RuVector for reinforcement learning,
pattern discovery, and self-improving agents.

Bridges Agent Lightning concepts with AgentDB's 9 RL algorithms and RuVector's
GNN-enhanced vector search for continuous agent improvement.
"""

import asyncio
import json
import os
import subprocess
from typing import Any, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agentic-lightning")

# Configuration
AGENTDB_PATH = os.environ.get("AGENTDB_PATH", "./agentdb.db")
DEFAULT_ALGO = os.environ.get("LIGHTNING_DEFAULT_ALGO", "ppo")

# Session cache (in-memory tracking)
_sessions: dict[str, dict] = {}


def _run_agentdb_cli(command: str, args: dict) -> dict:
    """Execute agentdb CLI command and return parsed result."""
    try:
        cmd_args = ["npx", "agentdb", command]
        for key, value in args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(f"--{key}")
                else:
                    cmd_args.extend([f"--{key}", str(value)])

        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(AGENTDB_PATH) or "."
        )

        if result.returncode != 0:
            return {"error": result.stderr or "Command failed", "success": False}

        # Try to parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"output": result.stdout, "success": True}

    except subprocess.TimeoutExpired:
        return {"error": "Command timed out", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def _call_agentdb_mcp(tool: str, params: dict) -> dict:
    """Call AgentDB MCP tool via CLI bridge."""
    try:
        # Use npx to call agentdb's MCP tools
        cmd = [
            "npx", "agentdb", "mcp-call",
            "--tool", tool,
            "--params", json.dumps(params)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except:
                return {"output": result.stdout, "success": True}
        else:
            # Fallback: direct Node.js execution
            return _direct_node_call(tool, params)

    except Exception as e:
        return {"error": str(e), "success": False}


def _direct_node_call(tool: str, params: dict) -> dict:
    """Direct Node.js call to AgentDB."""
    js_code = f"""
    const {{ AgentDB, LearningSystem, ReasoningBank, EmbeddingService }} = await import('agentdb');
    const db = new AgentDB('{AGENTDB_PATH}');
    await db.initialize();
    const result = await db.{tool}({json.dumps(params)});
    console.log(JSON.stringify(result));
    """

    try:
        result = subprocess.run(
            ["node", "--input-type=module", "-e", js_code],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"error": result.stderr, "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# Session Management Tools
# =============================================================================

@mcp.tool()
async def lightning_start_session(
    user_id: str,
    session_type: str = "ppo",
    learning_rate: float = 0.001,
    discount_factor: float = 0.99,
    exploration_rate: float = 0.1
) -> str:
    """
    Start RL training session with algorithm selection.

    Args:
        user_id: Identifier for the agent/user
        session_type: RL algorithm (q-learning, sarsa, dqn, policy-gradient,
                     actor-critic, ppo, decision-transformer, mcts, model-based)
        learning_rate: Learning rate alpha (0.0001-0.1)
        discount_factor: Discount factor gamma (0.9-0.999)
        exploration_rate: Epsilon for exploration (0.0-1.0)

    Returns:
        Session ID and configuration
    """
    valid_types = [
        "q-learning", "sarsa", "dqn", "policy-gradient",
        "actor-critic", "ppo", "decision-transformer", "mcts", "model-based"
    ]

    if session_type not in valid_types:
        return json.dumps({
            "error": f"Invalid session_type. Must be one of: {valid_types}",
            "success": False
        })

    result = _call_agentdb_mcp("learning_start_session", {
        "userId": user_id,
        "sessionType": session_type,
        "config": {
            "learningRate": learning_rate,
            "discountFactor": discount_factor,
            "explorationRate": exploration_rate
        }
    })

    if "sessionId" in result or "session_id" in result:
        session_id = result.get("sessionId") or result.get("session_id")
        _sessions[session_id] = {
            "userId": user_id,
            "sessionType": session_type,
            "startTime": datetime.now().isoformat(),
            "config": {
                "learningRate": learning_rate,
                "discountFactor": discount_factor,
                "explorationRate": exploration_rate
            }
        }

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_end_session(session_id: str) -> str:
    """
    End RL session and save final policy.

    Args:
        session_id: Session to end

    Returns:
        Session summary with training statistics
    """
    result = _call_agentdb_mcp("learning_end_session", {
        "sessionId": session_id
    })

    if session_id in _sessions:
        _sessions[session_id]["endTime"] = datetime.now().isoformat()
        _sessions[session_id]["status"] = "completed"

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_list_sessions(status: str = "all") -> str:
    """
    List RL training sessions.

    Args:
        status: Filter by status (active, completed, all)

    Returns:
        List of sessions with metadata
    """
    sessions_list = []
    for sid, data in _sessions.items():
        if status == "all":
            sessions_list.append({"sessionId": sid, **data})
        elif status == "active" and "endTime" not in data:
            sessions_list.append({"sessionId": sid, **data})
        elif status == "completed" and "endTime" in data:
            sessions_list.append({"sessionId": sid, **data})

    return json.dumps({
        "sessions": sessions_list,
        "count": len(sessions_list),
        "status_filter": status
    }, indent=2)


# =============================================================================
# Training & Prediction Tools
# =============================================================================

@mcp.tool()
async def lightning_predict(
    session_id: str,
    state: str
) -> str:
    """
    Get action prediction with confidence scores.

    Args:
        session_id: Active session ID
        state: Current state representation

    Returns:
        Recommended action with confidence, Q-value, and alternatives
    """
    result = _call_agentdb_mcp("learning_predict", {
        "sessionId": session_id,
        "state": state
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_feedback(
    session_id: str,
    state: str,
    action: str,
    reward: float,
    success: bool,
    next_state: Optional[str] = None
) -> str:
    """
    Submit feedback for RL learning.

    Args:
        session_id: Active session ID
        state: State where action was taken
        action: Action that was taken
        reward: Reward received (0.0-1.0)
        success: Whether action succeeded
        next_state: Resulting state (optional)

    Returns:
        Confirmation and updated policy statistics
    """
    result = _call_agentdb_mcp("learning_feedback", {
        "sessionId": session_id,
        "state": state,
        "action": action,
        "reward": reward,
        "success": success,
        "nextState": next_state,
        "timestamp": int(datetime.now().timestamp() * 1000)
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_train(
    session_id: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> str:
    """
    Batch training on collected experiences.

    Args:
        session_id: Session to train
        epochs: Number of training epochs
        batch_size: Batch size for updates
        learning_rate: Learning rate for this training run

    Returns:
        Training results with loss, reward, and convergence metrics
    """
    result = _call_agentdb_mcp("learning_train", {
        "sessionId": session_id,
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_train_gnn(
    epochs: int = 50,
    batch_size: int = 32
) -> str:
    """
    Train GNN model on collected pattern samples.

    Requires RuVector GNN backend. Uses accumulated pattern outcomes
    to improve semantic search through graph neural network learning.

    Args:
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        GNN training results with loss curve
    """
    # This calls ReasoningBank.trainGNN via AgentDB
    result = _call_agentdb_mcp("reasoning_train_gnn", {
        "epochs": epochs,
        "batchSize": batch_size
    })

    return json.dumps(result, indent=2)


# =============================================================================
# Pattern Management Tools
# =============================================================================

@mcp.tool()
async def lightning_store_pattern(
    task_type: str,
    approach: str,
    success_rate: float = 0.5,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None
) -> str:
    """
    Store reasoning pattern for semantic retrieval.

    Args:
        task_type: Type of task (e.g., 'code_review', 'test_generation')
        approach: Description of the reasoning approach
        success_rate: Initial success rate (0.0-1.0)
        tags: Tags for categorization
        metadata: Additional context

    Returns:
        Pattern ID and embedding info
    """
    result = _call_agentdb_mcp("agentdb_pattern_store", {
        "taskType": task_type,
        "approach": approach,
        "successRate": success_rate,
        "tags": tags or [],
        "metadata": metadata or {}
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_search_patterns(
    task: str,
    k: int = 5,
    use_gnn: bool = False,
    threshold: float = 0.0,
    task_type_filter: Optional[str] = None,
    min_success_rate: Optional[float] = None
) -> str:
    """
    Search patterns by semantic similarity with optional GNN enhancement.

    Args:
        task: Query task description
        k: Number of patterns to return
        use_gnn: Enable GNN-enhanced search (requires @ruvector/gnn)
        threshold: Minimum similarity threshold
        task_type_filter: Filter by task type
        min_success_rate: Filter by minimum success rate

    Returns:
        Ranked patterns with similarity scores
    """
    filters = {}
    if task_type_filter:
        filters["taskType"] = task_type_filter
    if min_success_rate is not None:
        filters["minSuccessRate"] = min_success_rate

    result = _call_agentdb_mcp("agentdb_pattern_search", {
        "task": task,
        "k": k,
        "useGNN": use_gnn,
        "threshold": threshold,
        "filters": filters if filters else None
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_record_outcome(
    pattern_id: int,
    success: bool,
    reward: Optional[float] = None
) -> str:
    """
    Record pattern outcome for continuous learning.

    Updates pattern statistics and adds training sample to GNN buffer.

    Args:
        pattern_id: Pattern that was used
        success: Whether pattern led to success
        reward: Optional reward value (defaults to 1.0/0.0 based on success)

    Returns:
        Updated pattern statistics
    """
    # This maps to ReasoningBank.recordOutcome
    result = _call_agentdb_mcp("reasoning_record_outcome", {
        "patternId": pattern_id,
        "success": success,
        "reward": reward
    })

    return json.dumps(result, indent=2)


# =============================================================================
# Transfer Learning Tools
# =============================================================================

@mcp.tool()
async def lightning_transfer(
    source_session: Optional[str] = None,
    target_session: Optional[str] = None,
    source_task: Optional[str] = None,
    target_task: Optional[str] = None,
    min_similarity: float = 0.7,
    transfer_type: str = "all",
    max_transfers: int = 10
) -> str:
    """
    Transfer learning between sessions or tasks.

    Args:
        source_session: Source session ID
        target_session: Target session ID
        source_task: Source task pattern (alternative to session)
        target_task: Target task pattern (alternative to session)
        min_similarity: Minimum similarity for transfer
        transfer_type: What to transfer (all, episodes, skills)
        max_transfers: Maximum items to transfer

    Returns:
        Transfer results with counts and details
    """
    result = _call_agentdb_mcp("learning_transfer", {
        "sourceSession": source_session,
        "targetSession": target_session,
        "sourceTask": source_task,
        "targetTask": target_task,
        "minSimilarity": min_similarity,
        "transferType": transfer_type,
        "maxTransfers": max_transfers
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_explain(
    query: str,
    k: int = 5,
    explain_depth: str = "detailed",
    include_causal: bool = True
) -> str:
    """
    XAI explanations for action recommendations.

    Args:
        query: Query to explain recommendations for
        k: Number of recommendations to explain
        explain_depth: Level of detail (summary, detailed, full)
        include_causal: Include causal reasoning chains

    Returns:
        Explainable recommendations with evidence and reasoning
    """
    result = _call_agentdb_mcp("learning_explain", {
        "query": query,
        "k": k,
        "explainDepth": explain_depth,
        "includeCausal": include_causal,
        "includeConfidence": True,
        "includeEvidence": True
    })

    return json.dumps(result, indent=2)


# =============================================================================
# Metrics & Analysis Tools
# =============================================================================

@mcp.tool()
async def lightning_metrics(
    session_id: Optional[str] = None,
    time_window_days: int = 7,
    include_trends: bool = True,
    group_by: str = "task"
) -> str:
    """
    Get learning performance metrics.

    Args:
        session_id: Specific session (optional, all sessions if not provided)
        time_window_days: Time window for analysis
        include_trends: Include daily trend data
        group_by: Group metrics by (task, session)

    Returns:
        Comprehensive metrics with trends and policy improvement
    """
    result = _call_agentdb_mcp("learning_metrics", {
        "sessionId": session_id,
        "timeWindowDays": time_window_days,
        "includeTrends": include_trends,
        "groupBy": group_by
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_convergence(session_id: str) -> str:
    """
    Calculate policy convergence rate.

    Analyzes Q-value changes across policy versions to determine
    if training has converged.

    Args:
        session_id: Session to analyze

    Returns:
        Convergence rate (0-1, higher = more converged)
    """
    # Convergence is calculated from policy version history
    result = _call_agentdb_mcp("learning_convergence", {
        "sessionId": session_id
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_nightly_learn(
    min_sample_size: int = 30,
    confidence_threshold: float = 0.6,
    auto_experiments: bool = True
) -> str:
    """
    Run nightly learner for automated causal discovery.

    Discovers causal edges, runs A/B experiments, prunes low-confidence
    edges, and generates recommendations.

    Args:
        min_sample_size: Minimum samples for edge discovery
        confidence_threshold: Minimum confidence for edges
        auto_experiments: Create new experiments automatically

    Returns:
        Nightly learner report with discovered edges and recommendations
    """
    result = _call_agentdb_mcp("nightly_learn", {
        "minSampleSize": min_sample_size,
        "confidenceThreshold": confidence_threshold,
        "autoExperiments": auto_experiments
    })

    return json.dumps(result, indent=2)


# =============================================================================
# Integration Tools
# =============================================================================

@mcp.tool()
async def lightning_record_tool_execution(
    session_id: str,
    tool_name: str,
    action: str,
    outcome: str,
    reward: float,
    success: bool,
    latency_ms: Optional[int] = None,
    state_before: Optional[dict] = None,
    state_after: Optional[dict] = None
) -> str:
    """
    Record tool execution as RL experience for offline learning.

    Use this to record Claude tool executions as learning experiences.

    Args:
        session_id: Active session ID
        tool_name: Name of tool executed
        action: Action taken (tool parameters summary)
        outcome: Outcome description
        reward: Reward value
        success: Whether tool succeeded
        latency_ms: Execution latency
        state_before: State before execution
        state_after: State after execution

    Returns:
        Experience ID
    """
    result = _call_agentdb_mcp("learning_record_experience", {
        "sessionId": session_id,
        "toolName": tool_name,
        "action": action,
        "outcome": outcome,
        "reward": reward,
        "success": success,
        "latencyMs": latency_ms,
        "stateBefore": state_before,
        "stateAfter": state_after
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_calculate_reward(
    success: bool,
    target_achieved: bool = True,
    efficiency_score: float = 0.5,
    quality_score: float = 0.5,
    time_taken_ms: Optional[int] = None,
    expected_time_ms: Optional[int] = None,
    reward_function: str = "standard",
    include_causal: bool = True,
    episode_id: Optional[int] = None
) -> str:
    """
    Calculate shaped reward signal.

    Args:
        success: Whether action succeeded
        target_achieved: Whether target was reached
        efficiency_score: Efficiency metric (0-1)
        quality_score: Quality metric (0-1)
        time_taken_ms: Actual time taken
        expected_time_ms: Expected time
        reward_function: Function type (sparse, dense, shaped, standard)
        include_causal: Include causal impact adjustment
        episode_id: Episode for causal lookup

    Returns:
        Calculated reward (0-1)
    """
    result = _call_agentdb_mcp("learning_calculate_reward", {
        "success": success,
        "targetAchieved": target_achieved,
        "efficiencyScore": efficiency_score,
        "qualityScore": quality_score,
        "timeTakenMs": time_taken_ms,
        "expectedTimeMs": expected_time_ms,
        "rewardFunction": reward_function,
        "includeCausal": include_causal,
        "episodeId": episode_id
    })

    return json.dumps(result, indent=2)


@mcp.tool()
async def lightning_ruvector_status() -> str:
    """
    Get RuVector backend status and GNN state.

    Returns:
        Backend configuration, GNN initialization status, and performance metrics
    """
    # Check RuVector and GNN status
    result = {
        "ruvector": {
            "installed": True,
            "version": "0.1.24+"
        },
        "gnn": {
            "available": False,
            "initialized": False,
            "config": None
        },
        "agentdb": {
            "path": AGENTDB_PATH,
            "version": "2.0.0-alpha"
        }
    }

    # Try to check actual GNN status
    try:
        check_result = subprocess.run(
            ["node", "-e", "require('@ruvector/gnn'); console.log('OK')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if check_result.returncode == 0:
            result["gnn"]["available"] = True
    except:
        pass

    return json.dumps(result, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
