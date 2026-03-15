#!/bin/bash
# ============================================================================
# TurboFlow V4.0 Statusline — Powerline 3-Line Display
# ============================================================================
# Reads JSON from Claude Code statusline hook via stdin
# Displays: project, model, git, tokens, context %, cost, time, lines changed
# ============================================================================

INPUT=$(cat)

# Colors
BG_DEEP="\033[48;5;17m"
BG_DARK="\033[48;5;54m"
FG_MAGENTA="\033[38;5;201m"
FG_CYAN="\033[38;5;51m"
FG_GREEN="\033[38;5;82m"
FG_YELLOW="\033[38;5;226m"
FG_PINK="\033[38;5;198m"
FG_BLUE="\033[38;5;33m"
FG_ORANGE="\033[38;5;214m"
FG_RED="\033[38;5;196m"
FG_WHITE="\033[38;5;255m"
FG_GRAY="\033[38;5;244m"
BG_MAGENTA="\033[48;5;201m"
BG_CYAN="\033[48;5;51m"
BG_GREEN="\033[48;5;82m"
BG_YELLOW="\033[48;5;226m"
BG_PINK="\033[48;5;198m"
BG_BLUE="\033[48;5;33m"
BG_RED="\033[48;5;196m"
RST="\033[0m"
BOLD="\033[1m"
SEP=""

# Extract fields from JSON input
MODEL=$(echo "$INPUT" | jq -r '.model.display_name // "Claude"' 2>/dev/null)
VERSION=$(echo "$INPUT" | jq -r '.version // ""' 2>/dev/null)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // ""' 2>/dev/null | cut -c1-8)
CWD=$(echo "$INPUT" | jq -r '.workspace.current_dir // .cwd // "~"' 2>/dev/null)
PROJECT_NAME=$(basename "$CWD" 2>/dev/null || echo "project")
COST_USD=$(echo "$INPUT" | jq -r '.cost.total_cost_usd // 0' 2>/dev/null)
DURATION_MS=$(echo "$INPUT" | jq -r '.cost.total_duration_ms // 0' 2>/dev/null)
LINES_ADDED=$(echo "$INPUT" | jq -r '.cost.total_lines_added // 0' 2>/dev/null)
LINES_REMOVED=$(echo "$INPUT" | jq -r '.cost.total_lines_removed // 0' 2>/dev/null)
CTX_INPUT=$(echo "$INPUT" | jq -r '.context_window.total_input_tokens // 0' 2>/dev/null)
CTX_OUTPUT=$(echo "$INPUT" | jq -r '.context_window.total_output_tokens // 0' 2>/dev/null)
CTX_SIZE=$(echo "$INPUT" | jq -r '.context_window.context_window_size // 200000' 2>/dev/null)
CTX_USED_PCT=$(echo "$INPUT" | jq -r '.context_window.used_percentage // 0' 2>/dev/null | cut -d. -f1)
CACHE_CREATE=$(echo "$INPUT" | jq -r '.context_window.current_usage.cache_creation_input_tokens // 0' 2>/dev/null)
CACHE_READ=$(echo "$INPUT" | jq -r '.context_window.current_usage.cache_read_input_tokens // 0' 2>/dev/null)

# Git info
GIT_BRANCH=""
GIT_DIRTY=""
if command -v git &>/dev/null && git -C "$CWD" rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    GIT_BRANCH=$(git -C "$CWD" branch --show-current 2>/dev/null || echo "")
    [[ -n $(git -C "$CWD" status --porcelain 2>/dev/null) ]] && GIT_DIRTY="*" || GIT_DIRTY=""
fi

# Format duration (ms -> human readable)
format_duration() {
    local ms=$1 secs=$((ms / 1000)) mins=$((ms / 60000)) hours=$((ms / 3600000))
    mins=$((mins % 60)); secs=$((secs % 60))
    if [ $hours -gt 0 ]; then echo "${hours}h${mins}m"
    elif [ $mins -gt 0 ]; then echo "${mins}m${secs}s"
    else echo "${secs}s"; fi
}

# Format token counts (1234567 -> 1M, 12345 -> 12k)
format_tokens() {
    local tokens=$1
    if [ $tokens -ge 1000000 ]; then echo "$((tokens / 1000000))M"
    elif [ $tokens -ge 1000 ]; then echo "$((tokens / 1000))k"
    else echo "$tokens"; fi
}

# Context usage bar (20 chars wide, color-coded)
ctx_bar() {
    local pct=$1 width=${2:-20} filled=$((pct * width / 100)) empty=$((width - filled))
    local bar_color="$FG_GREEN"
    [ $pct -ge 50 ] && bar_color="$FG_CYAN"
    [ $pct -ge 70 ] && bar_color="$FG_YELLOW"
    [ $pct -ge 85 ] && bar_color="$FG_ORANGE"
    [ $pct -ge 95 ] && bar_color="$FG_RED"
    printf "${bar_color}"; printf "%${filled}s" | tr ' ' '#'
    printf "${FG_GRAY}"; printf "%${empty}s" | tr ' ' '-'; printf "${RST}"
}

# Computed values
DURATION_FMT=$(format_duration $DURATION_MS)
CTX_TOTAL=$((CTX_INPUT + CTX_OUTPUT))
CTX_TOTAL_FMT=$(format_tokens $CTX_TOTAL)
CTX_SIZE_FMT=$(format_tokens $CTX_SIZE)
CACHE_HIT_PCT=0
[ $((CACHE_READ + CACHE_CREATE)) -gt 0 ] && CACHE_HIT_PCT=$((CACHE_READ * 100 / (CACHE_READ + CACHE_CREATE + 1)))
MODEL_ABBREV=$(echo "$MODEL" | sed 's/Sonnet/So/;s/Opus/Op/;s/Haiku/Ha/' | cut -c1-8)
COST_FMT=$(printf "%.2f" $COST_USD 2>/dev/null || echo "0.00")

# Line 1: Project name, Model, Git branch, version, TF 4.0, session ID
LINE1="${BG_MAGENTA}${FG_WHITE}${BOLD} [Project] ${PROJECT_NAME} ${RST}${FG_MAGENTA}${BG_CYAN}${SEP}${RST}${BG_CYAN}${FG_WHITE}${BOLD} [Model] ${MODEL_ABBREV} ${RST}${FG_CYAN}${BG_GREEN}${SEP}${RST}"
[ -n "$GIT_BRANCH" ] && LINE1+="${BG_GREEN}${FG_WHITE}${BOLD} [Git] ${GIT_BRANCH}${GIT_DIRTY} ${RST}${FG_GREEN}${BG_BLUE}${SEP}${RST}" || LINE1+="${BG_GREEN}${FG_WHITE}${BOLD} [Git] — ${RST}${FG_GREEN}${BG_BLUE}${SEP}${RST}"
[ -n "$VERSION" ] && LINE1+="${BG_BLUE}${FG_WHITE} [v] ${VERSION} ${RST}${FG_BLUE}${BG_PINK}${SEP}${RST}"
LINE1+="${BG_PINK}${FG_WHITE} [TF] 4.0 ${RST}"
[ -n "$SESSION_ID" ] && LINE1+="${FG_PINK}${BG_DEEP}${SEP}${RST}${BG_DEEP}${FG_GRAY} [SID] ${SESSION_ID} ${RST}"

# Line 2: Token usage, context bar (20 chars), cache hit %, cost, time
LINE2="${BG_YELLOW}${FG_WHITE}${BOLD} [Tokens] ${CTX_TOTAL_FMT}/${CTX_SIZE_FMT} ${RST}${FG_YELLOW}${BG_DARK}${SEP}${RST}${BG_DARK}${FG_WHITE} [Ctx] $(ctx_bar $CTX_USED_PCT 20) ${CTX_USED_PCT}% ${RST}${FG_DARK}${BG_CYAN}${SEP}${RST}"
[ $CACHE_HIT_PCT -gt 0 ] && LINE2+="${BG_CYAN}${FG_WHITE} [Cache] ${CACHE_HIT_PCT}% ${RST}" || LINE2+="${BG_CYAN}${FG_WHITE} [Cache] cold ${RST}"
LINE2+="${FG_CYAN}${BG_PINK}${SEP}${RST}${BG_PINK}${FG_WHITE}${BOLD} [Cost] \$${COST_FMT} ${RST}${FG_PINK}${BG_DEEP}${SEP}${RST}${BG_DEEP}${FG_CYAN} [Time] ${DURATION_FMT} ${RST}"

# Line 3: Lines added/removed, READY indicator
LINE3=""
[ $LINES_ADDED -gt 0 ] && LINE3+="${BG_GREEN}${FG_WHITE} [+${LINES_ADDED}] ${RST}${FG_GREEN}${BG_RED}${SEP}${RST}${BG_RED}${FG_WHITE} [-${LINES_REMOVED}] ${RST}${FG_RED}${BG_BLUE}${SEP}${RST}" || LINE3+="${BG_BLUE}${FG_WHITE}"
LINE3+="${BG_GREEN}${FG_WHITE}${BOLD} [READY] ${RST}${FG_GREEN}${RST}"

echo -e "${LINE1}"
echo -e "${LINE2}"
echo -e "${LINE3}"
