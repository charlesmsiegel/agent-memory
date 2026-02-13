# pyclawmem

Memory, personality, and context continuity for LLM agents. Python implementation inspired by OpenClaw's architecture, using AWS Bedrock.

## Quick Start

```bash
pip install -e ./pyclawmem
```

```python
from pyclawmem import AgentMemory

agent = AgentMemory.from_config("path/to/CONFIG.yaml")
session = agent.start_session()

# Your message loop:
prepared = agent.before_llm_call("What's the status of the project?", session)
response = your_llm_call(prepared.messages, prepared.system_prompt)
agent.after_llm_call(response, session)

# When done:
agent.end_session(session)
agent.close()
```

`before_llm_call` handles auto-recall, context pruning, and compaction automatically.
`after_llm_call` persists the response to the session transcript.
`end_session` auto-captures facts and exports the session to markdown.

### Chainlit Example

```python
import chainlit as cl
from pyclawmem import AgentMemory

agent = AgentMemory.from_config("CONFIG.yaml")

@cl.on_chat_start
async def start():
    session = agent.start_session()
    cl.user_session.set("session", session)
    await cl.Message(content=f"Hi! I'm {agent.identity.name}.").send()

@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("session")
    prepared = agent.before_llm_call(message.content, session)

    # Call your LLM (Bedrock, OpenAI, etc.)
    response = await call_your_llm(prepared.messages, prepared.system_prompt)

    agent.after_llm_call(response, session)
    await cl.Message(content=response).send()

@cl.on_chat_end
async def end():
    session = cl.user_session.get("session")
    if session:
        agent.end_session(session)
```

### Properties

```python
agent.identity        # AssistantIdentity(name, avatar, emoji)
agent.message_prefix  # "[AgentName]" for outbound messages
agent.context_tokens  # Effective context window size
```

### Direct Access

```python
agent.search_memory("authentication flow")  # File-based hybrid search
agent.search_facts("user preferences")      # Conversational fact search
agent.store_fact("User prefers dark mode")   # Manual fact storage
agent.forget_fact("abc123")                  # GDPR deletion
agent.resolve_typing_mode(is_group_chat=True)
```

## Architecture

```
pyclawmem/
├── CONFIG.yaml              # Model IDs, regions, tuning parameters
├── memory/
│   ├── types.py             # Pydantic data models
│   ├── embeddings.py        # Bedrock Titan embeddings
│   ├── store.py             # SQLite + FTS5 storage
│   └── manager.py           # Hybrid search, file indexing, sync
├── personality/
│   ├── soul.py              # SOUL.md loading + system prompt
│   ├── identity.py          # IDENTITY.md parsing
│   └── workspace.py         # Bootstrap file management
├── context/
│   ├── session.py           # JSONL transcript persistence
│   ├── compaction.py        # LLM summarization when context fills
│   └── pruning.py           # Stale context removal
└── integration.py           # Example: how to wire into an agent loop
```

## Dependencies

```
boto3       # AWS Bedrock API calls
numpy       # Vector normalization and cosine similarity
pydantic    # Data models and validation
pyyaml      # CONFIG.yaml parsing
```

Install: `pip install boto3 numpy pydantic pyyaml`

## Configuration

Edit `CONFIG.yaml` to set:

- **Embedding model**: Bedrock Titan model ID and dimensions
- **LLM model**: Bedrock Claude model for compaction summaries
- **Memory**: SQLite path, chunk size, hybrid search weights
- **Context**: Token budget, compaction threshold
- **Workspace**: Bootstrap files directory

AWS credentials are resolved via standard boto3 chain (env vars, `~/.aws/credentials`, IAM role).

## How It Works

### Memory (Hybrid Search)

Files in the workspace `memory/` directory are chunked, embedded via Bedrock Titan, and stored in SQLite. Search combines:

- **Vector similarity** (cosine, 70% weight) — finds semantically related content
- **BM25 keyword search** (FTS5, 30% weight) — finds exact term matches

Results are merged, deduped by (path, line), and scored.

### Personality

Workspace bootstrap files define the agent's identity:

| File | Purpose |
|------|---------|
| `SOUL.md` | Persona, values, behavioral philosophy |
| `IDENTITY.md` | Structured: name, emoji, creature, vibe, avatar |
| `USER.md` | Who the human is |
| `AGENTS.md` | Operating manual |
| `TOOLS.md` | Environment-specific notes |
| `MEMORY.md` | Curated long-term memory |

These are loaded into the system prompt as "Project Context". SOUL.md gets special treatment: the system prompt explicitly instructs the LLM to embody its persona.

### Context Continuity

- **Session persistence**: JSONL transcripts with auto-repair for corrupted files
- **Compaction**: When context exceeds 85% of budget, older messages are LLM-summarized. Tool failures and file operations are preserved across compaction.
- **Pruning**: Stale context (old identity reads, expired memory injections) is removed before each LLM call.

## Connection Points

To integrate pyclawmem into your agent, hook into these lifecycle events:

### Session Start
```python
from pyclawmem.personality.workspace import load_bootstrap_files, build_context_prompt
from pyclawmem.memory.manager import MemoryManager

# Build system prompt from personality files
files = load_bootstrap_files("~/.pyclawmem/workspace")
system_prompt = build_context_prompt(files)

# Sync memory index
memory.sync(["~/.pyclawmem/workspace/memory"])
```

### Before Each LLM Call
```python
from pyclawmem.context.pruning import prune_stale_context
from pyclawmem.context.compaction import needs_compaction, compact, apply_compaction

# Auto-recall: search memory for relevant context
results = memory.search(user_message)
# Inject results as <relevant-memories> in system context

# Prune stale context
messages = prune_stale_context(messages)

# Compact if needed
if needs_compaction(messages, max_tokens):
    result = compact(messages, model_id="anthropic.claude-3-haiku-20240307-v1:0")
    messages = apply_compaction(messages, result)
```

### After Each LLM Call
```python
from pyclawmem.context.session import SessionStore

# Persist to transcript
sessions.append_message(session, assistant_message)
```

### Session End
```python
memory.close()
```

## What This Does NOT Include

This is a **library**, not a complete agent. You still need:

- An LLM call layer (Bedrock Converse, LangChain, etc.)
- A user interface (CLI, web, messaging platform)
- Tool execution framework
- Authentication and access control

pyclawmem provides the memory, personality, and context management that sits between your agent loop and your LLM provider.
