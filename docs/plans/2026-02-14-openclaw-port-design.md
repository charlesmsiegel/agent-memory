# OpenClaw Feature Port — Design Document

**Date:** 2026-02-14
**Status:** Approved

## Goal

Port missing OpenClaw features into `src/agent_memory/`, adding multi-provider
embeddings (Bedrock, OpenRouter, local GGUF), a provider fallback chain, and
full application-level features (citations, prompt modes, bootstrap truncation,
agent config, sandbox, group chat, heartbeat scheduler).

## Approach

Bottom-up layering (Approach A): build the abstract embedding interface first,
then layer features on top in dependency order.

---

## Section 1: Embedding Provider Abstraction

### Protocol

A `typing.Protocol` in `memory/embeddings.py`:

```python
class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

### Implementations

| Provider | Module | Dependencies | Notes |
|---|---|---|---|
| `BedrockEmbeddings` | `embeddings_bedrock.py` | `boto3` (existing) | Extracted from current `embeddings.py`, unchanged logic |
| `OpenRouterEmbeddings` | `embeddings_openrouter.py` | `httpx` (new core dep) | OpenAI-compatible `/api/v1/embeddings`, native batch support |
| `LocalGGUFEmbeddings` | `embeddings_gguf.py` | `llama-cpp-python` (optional dep) | In-process, explicit `model_path` required (no auto-download), dimensions auto-detected via `model.n_embd()` |

### FallbackChain

```python
class FallbackChain:
    """Tries providers in order. Falls back on any exception."""
    def __init__(self, providers: list[EmbeddingProvider]): ...
    # Implements EmbeddingProvider protocol
    # Exposes .active_provider for introspection
```

### Dependency Strategy

- `boto3` — stays in core deps
- `httpx` — added to core deps
- `llama-cpp-python` — optional: `[project.optional-dependencies] local = ["llama-cpp-python>=0.3"]`
- `LocalGGUFEmbeddings.__init__` raises `ImportError` if llama-cpp-python missing

### LocalGGUFEmbeddings Details

- `__init__(self, model_path: str)` — path must exist on disk, raises `FileNotFoundError` otherwise
- Dimensions auto-detected from model via `model.n_embd()` after loading
- No `dimensions` config field needed for GGUF

---

## Section 2: Config Schema + Agent Config

### Expanded CONFIG.yaml

```yaml
embedding:
  provider: bedrock                # bedrock | openrouter | gguf
  fallback: openrouter             # optional: try this if primary fails

  # Bedrock-specific
  model_id: amazon.titan-embed-text-v2:0
  dimensions: 1024
  region: us-east-1

  # OpenRouter-specific
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
    model: thenlper/gte-large
    dimensions: 1024

  # GGUF-specific
  gguf:
    model_path: /path/to/model.gguf

llm:
  provider: bedrock
  model_id: us.anthropic.claude-haiku-4-5-20251001-v1:0
  region: us-east-1
  max_tokens: 1024

memory:
  db_path: ~/.pyclawmem/memory.sqlite
  facts_db_path: ~/.pyclawmem/facts.sqlite
  memory_dir: ~/.pyclawmem/workspace/memory
  chunk_tokens: 400
  chunk_overlap: 80
  hybrid_vector_weight: 0.7
  hybrid_text_weight: 0.3
  max_results: 6
  min_score: 0.35
  citations: auto                  # auto | on | off

context:
  max_context_tokens: 100000
  compaction_threshold: 0.85
  summary_preserve: "decisions, TODOs, open questions, constraints"
  session_dir: ~/.pyclawmem/sessions
  prompt_mode: full                # full | minimal | none
  bootstrap_max_chars: 20000       # per-file truncation limit

workspace:
  dir: ~/.pyclawmem/workspace

agent:
  id: default
  name: null

  sandbox:
    mode: "off"                    # off | non-main | all
    workspace_access: rw           # none | ro | rw

  group_chat:
    enabled: false
    respond_to_mentions: true
    respond_to_direct: true
    quiet_unless_mentioned: false

  heartbeat:
    enabled: false
    interval_minutes: 15
    quiet_hours:
      start: "22:00"
      end: "07:00"
    tasks: []
```

### AgentConfig Dataclass

New file `src/agent_memory/config.py`:

```python
@dataclass
class SandboxConfig:
    mode: Literal["off", "non-main", "all"] = "off"
    workspace_access: Literal["none", "ro", "rw"] = "rw"

@dataclass
class GroupChatConfig:
    enabled: bool = False
    respond_to_mentions: bool = True
    respond_to_direct: bool = True
    quiet_unless_mentioned: bool = False

@dataclass
class HeartbeatConfig:
    enabled: bool = False
    interval_minutes: int = 15
    quiet_start: str | None = None
    quiet_end: str | None = None
    tasks: list[str] = field(default_factory=list)

@dataclass
class AgentConfig:
    id: str = "default"
    name: str | None = None
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    group_chat: GroupChatConfig = field(default_factory=GroupChatConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
```

### from_config Changes

`resolve_embeddings(cfg)` reads the `embedding` section, builds the primary
provider, optionally wraps in `FallbackChain` if `fallback` is set.

---

## Section 3: Context Features

### Citations

`CitationMode` enum in `memory/types.py`:

```python
class CitationMode(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"
```

`MemorySearchResult` gets a `citation` property:

```python
@property
def citation(self) -> str:
    return f"{self.path}#L{self.start_line}-L{self.end_line}"
```

`auto` logic: show citations when `is_main_session=True`, hide in group chat.

### Prompt Modes

```python
class PromptMode(str, Enum):
    FULL = "full"
    MINIMAL = "minimal"
    NONE = "none"
```

- `full` — all bootstrap files, soul, daily logs (today's behavior)
- `minimal` — AGENTS.md + TOOLS.md only (subagents)
- `none` — one-liner: "You are {name}."
- `is_subagent=True` forces `minimal` regardless of config

### Bootstrap Truncation

`load_bootstrap_files` gets `max_chars_per_file` param (default 20,000).
Files exceeding the limit get `\n\n[... truncated at {limit} chars]` appended.

---

## Section 4: Application Features

### Sandbox

Application-level policy enforcement (no OS-level sandboxing):

- `start_session`: sandboxed sessions skip `ensure_workspace`, memory flush,
  session export
- `execute_tool`: `workspace_access == "ro"` blocks writes; `"none"` blocks
  reads and writes. Tool definitions filtered via `get_tool_definitions(sandboxed=...)`
- `mode == "all"`: always sandboxed. `"non-main"`: sandboxed when
  `is_main_session=False`. `"off"`: never sandboxed.

### Group Chat Config

- Wires `group_chat.enabled` into `resolve_typing_mode`
- `CitationMode.AUTO` uses it to decide citation visibility
- New `should_respond()` helper on facade:

```python
def should_respond(self, *, was_mentioned=False, is_direct=False) -> bool:
    gc = self._agent_config.group_chat
    if not gc.enabled:
        return True
    if gc.quiet_unless_mentioned and not was_mentioned:
        return False
    if is_direct and gc.respond_to_direct:
        return True
    if was_mentioned and gc.respond_to_mentions:
        return True
    return not gc.quiet_unless_mentioned
```

### Heartbeat Scheduler

New file `src/agent_memory/heartbeat.py`:

```python
class HeartbeatScheduler:
    def __init__(self, config, workspace_dir, user_timezone="UTC"): ...
    def is_quiet_hours(self) -> bool: ...
    def tasks_due(self) -> list[str]: ...
    def mark_completed(self, task: str) -> None: ...
    def parse_heartbeat_file(self) -> list[str]: ...
```

Caller-polled (no background thread). Facade exposes `self.heartbeat` property.

---

## Section 5: Facade Wiring

### resolve_embeddings Factory

```python
def resolve_embeddings(cfg: dict) -> EmbeddingProvider:
    primary = _build_provider(cfg, cfg["provider"])
    fallback_name = cfg.get("fallback")
    if fallback_name:
        fallback = _build_provider(cfg, fallback_name)
        return FallbackChain([primary, fallback])
    return primary
```

### Type Signature Changes

- `MemoryManager.__init__(..., embeddings: EmbeddingProvider, ...)`
- `FactStore.__init__(..., embeddings: EmbeddingProvider)`

---

## File Change Summary

### New Files

| File | Purpose |
|---|---|
| `memory/embeddings.py` | `EmbeddingProvider` protocol, `FallbackChain`, `resolve_embeddings`, `_normalize` |
| `memory/embeddings_bedrock.py` | Extracted `BedrockEmbeddings` |
| `memory/embeddings_openrouter.py` | `OpenRouterEmbeddings` via httpx |
| `memory/embeddings_gguf.py` | `LocalGGUFEmbeddings` via llama-cpp-python |
| `config.py` | `AgentConfig`, `SandboxConfig`, `GroupChatConfig`, `HeartbeatConfig` |
| `heartbeat.py` | `HeartbeatScheduler` |

### Modified Files

| File | Changes |
|---|---|
| `memory/manager.py` | Type hint → `EmbeddingProvider` |
| `memory/facts.py` | Type hint → `EmbeddingProvider` |
| `memory/types.py` | Add `CitationMode`, `PromptMode`, `citation` property |
| `personality/workspace.py` | Add `max_chars_per_file` truncation |
| `facade.py` | Wire everything: providers, agent config, sandbox, group chat, heartbeat, prompt modes, citations |
| `CONFIG.yaml` | Full expanded schema |
| `pyproject.toml` | Add `httpx` core dep, `llama-cpp-python` optional `[local]` extra |

### Not Porting

- QMD sidecar backend (external binary, OpenClaw-specific)
- Docker/browser sandbox (OS-level, platform-specific)
- Prompt cache-TTL awareness (provider-specific cache APIs)
