#!/usr/bin/env python3
"""Generate VisionFlow Wardley Map with competitor positioning.

Shows why Genesis/Custom-Built positioning is strategic advantage,
not a liability — by placing competitors in the commodity zone
where there's no differentiation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(20, 14))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Evolution', fontsize=13, color='#c9d1d9', fontweight='bold', labelpad=15)
ax.set_ylabel('Value Chain (visible to user → invisible infrastructure)', fontsize=13,
              color='#c9d1d9', fontweight='bold', labelpad=15)

# Evolution axis labels
for x, label in [(0.125, 'Genesis'), (0.375, 'Custom-Built'), (0.625, 'Product (+rental)'), (0.875, 'Commodity (+utility)')]:
    ax.text(x, -0.03, label, ha='center', va='top', fontsize=10, color='#8b949e',
            fontstyle='italic')

# Phase boundaries
for x in [0.25, 0.5, 0.75]:
    ax.axvline(x, color='#21262d', linewidth=1, linestyle='--', alpha=0.5)

# Value chain gridlines
for y in [0.25, 0.5, 0.75]:
    ax.axhline(y, color='#21262d', linewidth=0.5, linestyle=':', alpha=0.3)

# === STRATEGIC ADVANTAGE ZONE ===
advantage_zone = FancyBboxPatch(
    (0.02, 0.55), 0.38, 0.40,
    boxstyle="round,pad=0.02",
    facecolor='#e9456015', edgecolor='#e94560', linewidth=2, linestyle='--'
)
ax.add_patch(advantage_zone)
ax.text(0.21, 0.93, 'STRATEGIC ADVANTAGE ZONE', ha='center', va='center',
        fontsize=11, color='#e94560', fontweight='bold', fontstyle='italic')
ax.text(0.21, 0.90, 'No competitor occupies this space', ha='center', va='center',
        fontsize=9, color='#e9456099')

# === COMMODITY CLUSTER (where competitors live) ===
competitor_zone = FancyBboxPatch(
    (0.52, 0.55), 0.44, 0.38,
    boxstyle="round,pad=0.02",
    facecolor='#6b728015', edgecolor='#6b7280', linewidth=1.5, linestyle=':'
)
ax.add_patch(competitor_zone)
ax.text(0.74, 0.91, 'COMPETITOR CLUSTER', ha='center', va='center',
        fontsize=11, color='#6b7280', fontweight='bold', fontstyle='italic')
ax.text(0.74, 0.88, 'All platforms compete here — no differentiation', ha='center', va='center',
        fontsize=9, color='#6b728099')

# === COMPONENT PLOTTING ===

def plot_component(x, y, label, color, size=80, label_offset=(8, 5), fontsize=9, bold=False):
    ax.scatter(x, y, c=color, s=size, zorder=5, edgecolors='white', linewidth=0.5)
    weight = 'bold' if bold else 'normal'
    ax.annotate(label, (x, y), xytext=label_offset, textcoords='offset points',
                fontsize=fontsize, color=color, fontweight=weight, zorder=6)

def plot_competitor(x, y, label, fontsize=8):
    ax.scatter(x, y, c='#6b7280', s=50, zorder=4, marker='s', edgecolors='#4b5563', linewidth=0.5, alpha=0.7)
    ax.annotate(label, (x, y), xytext=(6, 3), textcoords='offset points',
                fontsize=fontsize, color='#9ca3af', fontstyle='italic', zorder=6)

def plot_evolution(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#e94560', lw=1.8, alpha=0.6))

# --- User anchors ---
plot_component(0.59, 0.97, 'User Need', '#c9d1d9', size=120, fontsize=11, bold=True, label_offset=(10, -5))
plot_component(0.20, 0.92, 'Hard Problems', '#c9d1d9', size=120, fontsize=11, bold=True, label_offset=(-90, -5))

# --- VisionFlow differentiators (Genesis/Custom — cyan) ---
plot_component(0.25, 0.82, 'VisionClaw', '#00d4ff', size=120, fontsize=11, bold=True, label_offset=(10, 5))
plot_component(0.18, 0.75, 'OWL 2 Reasoning', '#00d4ff', size=90, label_offset=(-110, -5))
plot_component(0.15, 0.70, 'GPU Semantic Physics', '#00d4ff', size=90, label_offset=(-130, 5))
plot_component(0.22, 0.78, 'Judgment Broker', '#00d4ff', size=90, label_offset=(-110, -8))
plot_component(0.28, 0.72, 'Immersive XR', '#00d4ff', size=80, label_offset=(-80, 8))
plot_component(0.35, 0.85, 'Governance UI', '#ff6b6b', size=90, label_offset=(8, 5))
plot_component(0.28, 0.85, 'Knowledge Graph', '#ff6b6b', size=90, label_offset=(-105, -8))
plot_component(0.30, 0.68, 'Agent Control Surface', '#00d4ff', size=80, label_offset=(8, -10))

# --- Platform components (Custom-Built — purple) ---
plot_component(0.35, 0.78, 'Agent Skills', '#ff6b6b', size=80, label_offset=(8, 5))
plot_component(0.38, 0.62, 'Agentbox', '#8b5cf6', size=100, fontsize=10, bold=True, label_offset=(8, 5))
plot_component(0.42, 0.58, 'Nostr Relay Mesh', '#8b5cf6', size=80, label_offset=(8, -10))

# --- Protocol infrastructure (Custom-Built — green) ---
plot_component(0.30, 0.52, 'solid-pod-rs', '#10b981', size=90, fontsize=10, bold=True, label_offset=(-80, -8))
plot_component(0.28, 0.48, 'DID:Nostr', '#10b981', size=80, label_offset=(-75, 5))
plot_component(0.32, 0.45, 'WAC Access Control', '#10b981', size=70, label_offset=(-120, -8))
plot_component(0.20, 0.42, 'Web Ledger Payments', '#10b981', size=70, label_offset=(-140, 5))

# --- Open standards (Product — amber) ---
plot_component(0.55, 0.38, 'MCP Protocol', '#f59e0b', size=60, label_offset=(8, 5))
plot_component(0.60, 0.35, 'Nix Flakes', '#f59e0b', size=60, label_offset=(8, -8))
plot_component(0.55, 0.30, 'Nostr Protocol', '#f59e0b', size=60, label_offset=(-100, -8))
plot_component(0.52, 0.27, 'Solid Protocol', '#f59e0b', size=60, label_offset=(-95, 5))

# --- Commodity infrastructure (grey) ---
plot_component(0.78, 0.22, 'CUDA', '#6b7280', size=50, label_offset=(8, -5))
plot_component(0.85, 0.18, 'Cloudflare Workers', '#6b7280', size=50, label_offset=(8, 5))
plot_component(0.87, 0.15, 'Docker', '#6b7280', size=50, label_offset=(8, -5))
plot_component(0.92, 0.12, 'PostgreSQL', '#6b7280', size=50, label_offset=(8, 5))
plot_component(0.90, 0.08, 'secp256k1', '#6b7280', size=50, label_offset=(-70, -5))

# === COMPETITORS (clustered in Product/Commodity, high value chain) ===
plot_competitor(0.70, 0.82, 'Google Spark')
plot_competitor(0.75, 0.78, 'OpenAI Codex')
plot_competitor(0.65, 0.76, 'Devin')
plot_competitor(0.72, 0.74, 'Google Jules')
plot_competitor(0.58, 0.80, 'OpenClaw')
plot_competitor(0.62, 0.72, 'Hermes')
plot_competitor(0.55, 0.68, 'Claude Code')
plot_competitor(0.68, 0.68, 'Cursor / Windsurf')
plot_competitor(0.60, 0.62, 'CrewAI')
plot_competitor(0.65, 0.60, 'AutoGen / MAF')
plot_competitor(0.58, 0.58, 'LangGraph')

# === EVOLUTION ARROWS ===
plot_evolution(0.18, 0.75, 0.35, 0.75)  # OWL 2 Reasoning evolving
plot_evolution(0.15, 0.70, 0.30, 0.70)  # GPU Semantic Physics evolving
plot_evolution(0.30, 0.68, 0.45, 0.68)  # Agent Control Surface evolving
plot_evolution(0.42, 0.58, 0.55, 0.58)  # Nostr Relay Mesh evolving
plot_evolution(0.28, 0.48, 0.42, 0.48)  # DID:Nostr evolving
plot_evolution(0.20, 0.42, 0.35, 0.42)  # Web Ledger Payments evolving

# === DEPENDENCY LINES (subtle) ===
deps = [
    (0.59, 0.97, 0.35, 0.85),  # User Need → Governance UI
    (0.59, 0.97, 0.28, 0.85),  # User Need → Knowledge Graph
    (0.59, 0.97, 0.35, 0.78),  # User Need → Agent Skills
    (0.20, 0.92, 0.22, 0.78),  # Hard Problems → Judgment Broker
    (0.20, 0.92, 0.28, 0.72),  # Hard Problems → Immersive XR
    (0.28, 0.85, 0.18, 0.75),  # Knowledge Graph → OWL 2 Reasoning
    (0.28, 0.85, 0.15, 0.70),  # Knowledge Graph → GPU Semantic Physics
    (0.35, 0.78, 0.38, 0.62),  # Agent Skills → Agentbox
    (0.22, 0.78, 0.30, 0.68),  # Judgment Broker → Agent Control Surface
    (0.30, 0.68, 0.42, 0.58),  # Agent Control Surface → Nostr Relay Mesh
    (0.18, 0.75, 0.25, 0.82),  # OWL 2 Reasoning → VisionClaw
    (0.15, 0.70, 0.78, 0.22),  # GPU Semantic Physics → CUDA
    (0.25, 0.82, 0.30, 0.52),  # VisionClaw → solid-pod-rs
    (0.38, 0.62, 0.30, 0.52),  # Agentbox → solid-pod-rs
    (0.38, 0.62, 0.60, 0.35),  # Agentbox → Nix Flakes
    (0.38, 0.62, 0.87, 0.15),  # Agentbox → Docker
    (0.30, 0.52, 0.28, 0.48),  # solid-pod-rs → DID:Nostr
    (0.30, 0.52, 0.32, 0.45),  # solid-pod-rs → WAC Access Control
    (0.28, 0.48, 0.90, 0.08),  # DID:Nostr → secp256k1
]
for x1, y1, x2, y2 in deps:
    ax.plot([x1, x2], [y1, y2], color='#30363d', linewidth=0.8, alpha=0.4, zorder=2)

# === LEGEND ===
legend_elements = [
    mpatches.Patch(facecolor='#00d4ff', edgecolor='white', label='Core Differentiators (VisionFlow only)'),
    mpatches.Patch(facecolor='#ff6b6b', edgecolor='white', label='User-Visible Capabilities'),
    mpatches.Patch(facecolor='#8b5cf6', edgecolor='white', label='Platform Components'),
    mpatches.Patch(facecolor='#10b981', edgecolor='white', label='Protocol Infrastructure'),
    mpatches.Patch(facecolor='#f59e0b', edgecolor='white', label='Open Standards'),
    mpatches.Patch(facecolor='#6b7280', edgecolor='white', label='Commodity Infrastructure'),
    mpatches.Patch(facecolor='#6b7280', edgecolor='#4b5563', label='Competitors (no formal reasoning,\nno crypto identity, no federation)'),
]
legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
                   facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9',
                   framealpha=0.9, borderpad=1, labelspacing=0.8)

# === TITLE ===
ax.set_title('VisionFlow Coordination Platform — Strategic Wardley Map',
             fontsize=16, color='#c9d1d9', fontweight='bold', pad=20)

# === ANNOTATION: Why Genesis = Advantage ===
ax.text(0.03, 0.58,
        'Genesis positioning = first-mover advantage.\n'
        'No competitor has formal reasoning, cryptographic\n'
        'agent identity, or cross-org federation.\n'
        'This is the moat.',
        fontsize=8, color='#e9456099', fontstyle='italic',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#e9456010', edgecolor='none'))

ax.tick_params(colors='#8b949e', which='both')
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/devuser/workspace/VisionFlow/assets/diagrams/wardley-map.png',
            dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.savefig('/home/devuser/workspace/project/docs/explanation/visionflow-wardley-map.png',
            dpi=150, facecolor='#0d1117', bbox_inches='tight')
print("Wardley maps saved successfully")
