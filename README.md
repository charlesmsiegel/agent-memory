# agent-memory
Agentic memory module, derived initially from translating openclaw's memory and personality module to python.

https://github.com/openclaw/openclaw

https://x.com/manthanguptaa/status/2015780646770323543?s=46&t=k51U7HkU6l1eaefXj3ArXA

https://snowan.gitbook.io/study-notes/ai-blogs/openclaw-memory-system-deep-dive

https://github.blog/ai-and-ml/github-copilot/building-an-agentic-memory-system-for-github-copilot/

Notes:


High-impact, practical

1. Memory importance scoring + decay — Not all memories are equal. A decision made today matters more than a casual remark from last month. Score memories by recency, access frequency, and explicit importance. Decay old, never-accessed memories so search results stay relevant. OpenClaw doesn'
t do this either — it's an open gap in both systems.

2. Cross-session conversation threading — Right now each session is isolated. If you discuss a project across 5 sessions, there's no explicit link between them. A thread ID or topic tag on sessions would let memory_search return results grouped by conversation thread, not just individual 
snippets.

3. Memory consolidation (automatic MEMORY.md curation) — The agent is told to review daily logs and update MEMORY.md during heartbeats, but there's no mechanism enforcing it. A periodic consolidation hook could: scan recent daily logs, extract key facts, deduplicate against MEMORY.md, and 
propose updates. Like how humans consolidate memories during sleep.

4. Chunking-aware of markdown structure — Our chunker splits by character count with overlap. It doesn't respect markdown headers, code blocks, or list boundaries. A chunk that splits a code block in half produces garbage embeddings. Structure-aware chunking (split at ## headers, keep code 
blocks intact) would improve search quality significantly.

5. Re-ranking after retrieval — Hybrid search returns candidates, but the scores are rough (cosine similarity + BM25 fusion). A lightweight re-ranker (Bedrock Cohere Rerank, or even an LLM-as-judge pass) on the top 20 candidates before returning top 6 would improve precision noticeably.

Medium-impact, architectural

6. Explicit memory graph — Facts are stored as flat entries. But "user prefers TypeScript" and "user is building Acme Dashboard" and "Acme Dashboard uses Next.js" form a graph. Linking related facts would let the agent traverse connections: "What do I know about the Acme project?" → pulls 
preferences, decisions, contacts, and tech stack in one query.

7. Multi-modal memory — We only index text. Images, PDFs, and code files uploaded through Chainlit get saved but only their text content (if any) is searchable. Adding image description (Bedrock Claude vision) and PDF extraction before indexing would make uploaded files actually useful.

8. Retention policies — Daily logs and session transcripts accumulate forever. After a year: ~365 daily logs, ~1000 session files, growing SQLite. Configurable retention (archive sessions older than 90 days, compress daily logs into monthly summaries) would keep the system performant long-
term.

9. Memory provenance — When the agent recalls a fact, it doesn't know when or why it was stored. Adding timestamps and source context ("stored from session on 2026-01-15, during API design discussion") would let the agent assess whether a memory is still current or potentially stale.

10. Conflict detection — If the user said "I prefer PostgreSQL" in January and "Let's use MongoDB" in February, both facts exist. The system has no way to detect the contradiction. A conflict detector on fact_store could flag when a new fact contradicts an existing one and prompt the agent to
resolve it.

Lower-impact, nice-to-have

11. Federated memory — Share curated memories across profiles. The "researcher" profile learns something useful for the "default" profile, but they're completely isolated. A shared facts layer (opt-in) would bridge them.

12. Memory export/import — Portable memory bundles for backup, migration, or sharing a personality setup with someone else.

13. Embedding model migration — Switching from Titan to a different model invalidates all cached embeddings (different vector space). A migration tool that re-embeds everything with the new model would make provider changes painless.
