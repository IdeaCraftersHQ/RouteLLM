#!/usr/bin/env bash
# Live smoke of the API dialect adapters with the real coding CLIs.
#
# Needs a running routellm server whose model/tier answers tool calls, and
# its caller key. Every CLI runs against throwaway config dirs and a
# throwaway project; nothing of the caller's real config or login is read.
#
#   ROUTELLM_BASE_URL=http://127.0.0.1:6060 \
#   ROUTELLM_KEY=<caller key> \
#   ROUTELLM_MODEL=<tier or router name> \
#   scripts/live_client_smoke.sh [claude claude-bedrock codex gemini opencode]
#
# Each CLI gets two headless turns: a plain answer, and one that must write
# hello.txt (containing `hi`) through a tool. Prints one PASS/FAIL line per
# turn. Exit 1 when any turn failed. Clients missing from PATH are skipped.
set -u

: "${ROUTELLM_BASE_URL:?set ROUTELLM_BASE_URL}"
: "${ROUTELLM_KEY:?set ROUTELLM_KEY}"
: "${ROUTELLM_MODEL:?set ROUTELLM_MODEL}"
LIMIT="${SMOKE_TIMEOUT:-180}"
BASE="${ROUTELLM_BASE_URL%/}"
KEY="$ROUTELLM_KEY"
MODEL="$ROUTELLM_MODEL"

ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT
PLAIN="Reply with one short sentence: what is 2+2?"
TOOL="Create a file hello.txt in the current directory containing exactly: hi. Then say done."
FAILED=0

# run_cli <name> <plain|tool> <cmd...>: runs in a fresh project dir.
run_cli() {
  local name="$1" kind="$2"
  shift 2
  local proj="$ROOT/$name/proj-$kind" out
  mkdir -p "$proj"
  out="$(cd "$proj" && "$@" </dev/null 2>&1)"
  local rc=$?
  local ok=0
  if [ "$kind" = plain ]; then
    [ $rc -eq 0 ] && printf '%s' "$out" | grep -q '4' && ok=1
  else
    # Small models sometimes nest the path; any hello.txt under the project counts.
    local f
    f="$(find "$proj" -name hello.txt -type f | head -1)"
    [ $rc -eq 0 ] && [ -n "$f" ] && grep -q 'hi' "$f" && ok=1
  fi
  if [ $ok -eq 1 ]; then
    echo "PASS $name $kind"
  else
    echo "FAIL $name $kind (exit $rc)"
    printf '%s\n' "$out" | tail -5 | sed 's/^/     /'
    FAILED=1
  fi
}

smoke_claude() {
  local mode="$1" cfg="$ROOT/claude-$1/cfg"
  mkdir -p "$cfg"
  local -a base=(env -u CLAUDE_CODE_OAUTH_TOKEN -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN
    -u ANTHROPIC_BASE_URL "CLAUDE_CONFIG_DIR=$cfg" "ANTHROPIC_MODEL=$MODEL"
    "ANTHROPIC_DEFAULT_OPUS_MODEL=$MODEL" "ANTHROPIC_DEFAULT_SONNET_MODEL=$MODEL"
    "ANTHROPIC_DEFAULT_HAIKU_MODEL=$MODEL" "ANTHROPIC_SMALL_FAST_MODEL=$MODEL"
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1)
  if [ "$mode" = bedrock ]; then
    base+=(CLAUDE_CODE_USE_BEDROCK=1 "ANTHROPIC_BEDROCK_BASE_URL=$BASE"
      "AWS_BEARER_TOKEN_BEDROCK=$KEY" AWS_REGION=us-east-1 CLAUDE_CODE_SKIP_BEDROCK_AUTH=1)
  else
    base+=("ANTHROPIC_BASE_URL=$BASE" "ANTHROPIC_AUTH_TOKEN=$KEY")
  fi
  run_cli "claude-$mode" plain "${base[@]}" timeout "$LIMIT" claude -p "$PLAIN" --tools ""
  run_cli "claude-$mode" tool "${base[@]}" timeout "$LIMIT" claude -p "$TOOL" \
    --tools Write --allowedTools Write
}

smoke_codex() {
  local home="$ROOT/codex/home"
  mkdir -p "$home"
  cat >"$home/config.toml" <<EOF
model = "$MODEL"
model_provider = "routellm"
approval_policy = "never"
sandbox_mode = "workspace-write"
web_search = "disabled"
[model_providers.routellm]
name = "routellm"
base_url = "$BASE/v1"
env_key = "ROUTELLM_KEY"
wire_api = "responses"
EOF
  local -a base=(env -i "HOME=$ROOT/codex" "PATH=$PATH" "CODEX_HOME=$home" "ROUTELLM_KEY=$KEY")
  run_cli codex plain "${base[@]}" timeout "$LIMIT" codex exec --skip-git-repo-check "$PLAIN"
  run_cli codex tool "${base[@]}" timeout "$LIMIT" codex exec --skip-git-repo-check "$TOOL"
}

smoke_gemini() {
  local home="$ROOT/gemini/home"
  mkdir -p "$home/.gemini"
  echo '{"security":{"auth":{"selectedType":"gemini-api-key"}}}' >"$home/.gemini/settings.json"
  local -a base=(env -i "HOME=$home" "PATH=$PATH" "GEMINI_CLI_HOME=$home"
    "GEMINI_API_KEY=$KEY" "GOOGLE_GEMINI_BASE_URL=$BASE")
  run_cli gemini plain "${base[@]}" timeout "$LIMIT" gemini -m "$MODEL" --skip-trust -p "$PLAIN"
  run_cli gemini tool "${base[@]}" timeout "$LIMIT" gemini -m "$MODEL" --skip-trust \
    --approval-mode auto_edit -p "$TOOL"
}

smoke_opencode() {
  local d="$ROOT/opencode"
  mkdir -p "$d/home" "$d/xdg/config" "$d/xdg/data" "$d/xdg/cache" "$d/xdg/state"
  cat >"$d/opencode.json" <<EOF
{
  "autoupdate": false,
  "share": "disabled",
  "provider": {
    "routellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "routellm",
      "options": {"baseURL": "$BASE/v1", "apiKey": "$KEY"},
      "models": {"$MODEL": {"name": "$MODEL", "tool_call": true}}
    }
  },
  "permission": {"edit": "allow", "bash": "deny", "webfetch": "deny"}
}
EOF
  local -a base=(env -i "HOME=$d/home" "PATH=$PATH" "XDG_CONFIG_HOME=$d/xdg/config"
    "XDG_DATA_HOME=$d/xdg/data" "XDG_CACHE_HOME=$d/xdg/cache" "XDG_STATE_HOME=$d/xdg/state"
    "OPENCODE_CONFIG=$d/opencode.json")
  run_cli opencode plain "${base[@]}" timeout "$LIMIT" opencode run --standalone --auto \
    -m "routellm/$MODEL" "$PLAIN"
  run_cli opencode tool "${base[@]}" timeout "$LIMIT" opencode run --standalone --auto \
    -m "routellm/$MODEL" "$TOOL"
}

want=("$@")
[ ${#want[@]} -eq 0 ] && want=(claude claude-bedrock codex gemini opencode)
for c in "${want[@]}"; do
  bin="${c%%-*}"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "SKIP $c ($bin not on PATH)"
    continue
  fi
  case "$c" in
    claude) smoke_claude messages ;;
    claude-bedrock) smoke_claude bedrock ;;
    codex) smoke_codex ;;
    gemini) smoke_gemini ;;
    opencode) smoke_opencode ;;
    *) echo "unknown client: $c" >&2; FAILED=1 ;;
  esac
done
exit $FAILED
