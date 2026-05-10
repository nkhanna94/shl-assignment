# Approach Document — SHL Assessment Recommender

## Problem & Design Choices

Hiring managers arrive with vague intent ("I need to assess a Java developer") and no knowledge of SHL's catalog vocabulary. The core challenge is bridging that gap through dialogue while staying strictly within the catalog.

**Architecture: stateless RAG agent**

Each `/chat` call receives the full conversation history. The pipeline is:
1. Concatenate all user turns → BM25 retrieval over catalog
2. Top-10 catalog items injected into system prompt as context
3. LLM (Groq / Llama-3.3-70b) generates reply + structured JSON recommendation block
4. Server validates every URL against catalog set before returning

No session state is stored server-side. All context lives in the request payload.

---

## Catalog & Retrieval

**Scraping:** Crawled `shl.com/solutions/products/product-catalog/` across 32 paginated pages (12 items/page) using `urllib` + regex. Extracted 377 Individual Test Solutions with name, URL, test types (A/B/C/D/E/K/P/S), remote testing flag, and adaptive/IRT flag. Pre-packaged Job Solutions filtered out. Product names HTML-unescaped at scrape time.

**Retrieval: BM25 + type-keyword boosting + role-context expansion + name-match bonus**

Initially used `sentence-transformers` (all-MiniLM-L6-v2) + FAISS for semantic search. Dropped it — PyTorch stack exceeded Render's 512MB free-tier RAM. Switched to `rank-bm25` (pure Python, <5MB RAM).

BM25 alone misses type intent ("personality test" → OPQ) and role intent ("analyst" → ability tests). Three layers compensate:

- **Role-context expansion:** each catalog item's BM25 document is enriched with domain synonyms for its test types (e.g., a type-P item also indexes "manager executive sales communication interpersonal team"). Bridges queries like "hiring a sales manager" to personality assessments without semantic embeddings.
- **Type-keyword boost:** if the query contains role/domain words (e.g., "personality", "analyst", "stakeholder"), items of the matching test type get a 1.5× BM25 score multiplier (once per item regardless of multi-type).
- **Name-match bonus:** each query token found directly in the item name adds a further 1.3× (capped at two matching tokens). Ensures `Java 8 (New)` outranks `Informatica (Developer)` for a "java developer" query even though both get the K-type boost.

After all boosts, items below 20% of the top score are dropped. If that leaves fewer than 5 results, the cutoff relaxes (20%→10%→5%→0%) until five items are returned — preventing over-pruning for narrow catalogs (e.g., only one Python-specific test exists).

**Why not semantic search?** For a 377-item structured catalog, BM25 + these three layers performs comparably, fits in 512MB, and adds zero latency.

---

## Prompt Design

System prompt enforces four behaviors via explicit numbered rules:

1. **Catalog-only URLs** — model instructed to use only URLs from injected context; server validates every URL against `CATALOG_URL_SET` as a hard backstop; hallucinated URLs silently dropped.
2. **Recommend vs. clarify logic** — explicit decision rules: recommend immediately if role + any one of (seniority / skill type / assessment type) is known; clarify only for truly vague queries; max 1–2 clarifying questions total.
3. **JD paste detection** — server detects "here is a text from job description" patterns and injects a CRITICAL override forcing immediate recommendations without clarification.
4. **Turn cap enforcement** — at turn 5+, system prompt augmented with urgency note ("X turns left"). At turn 7, server hard-overrides: top-8 BM25 results returned directly, bypassing LLM JSON parsing.
5. **Scope enforcement** — off-topic requests (hiring advice, legal, prompt injection) get a fixed refusal string with `recommendations: []`.

**Output format:** LLM appends a ` ```json {"recommend": [...], "done": bool} ``` ` block. Server extracts via regex (handles both code-fenced and raw JSON), strips it from user-facing reply. Fallback reply text never claims recommendations when the list is empty.

**Token budget:** k=10 retrieval candidates, terse catalog format (`Name [types] URL`), max_tokens=450. Keeps each call well under Groq's free daily limit (~350 tokens/call vs. ~1800 before).

**Model:** Llama-3.3-70b via Groq. Chosen for fastest free-tier inference (sub-2s p50) — critical given 30s timeout.

---

## Evaluation Approach

**Hard evals (schema compliance):**
Tested locally: vague query (no recs turn 1), JD paste (immediate recs), multi-turn with context (5–10 recs), refinement (updated shortlist), off-topic (empty recs + refusal), prompt injection (refusal), turn-7 fallback (forced recs), malformed LLM output (no 500).

**Recall@10:**
Model instructed to return 5–10 assessments. BM25 retrieves 10 boosted candidates; LLM picks best from those. Hard fallback at turn 7 returns top-8 BM25 results directly.

**Behavior probes:**
- Vague query turn 1 → `recommendations: []` ✓
- JD paste → immediate recommendations ✓
- Off-topic → fixed refusal, `recommendations: []` ✓
- Refinement ("add personality tests") → updated shortlist ✓
- Comparison ("difference between OPQ32r and MQM5") → grounded answer, `recommendations: []` ✓
- Malformed LLM output (None, empty, wrong types, bad URLs) → HTTP 200, safe fallback ✓

**What didn't work:**
- `sentence-transformers` — OOM on Render free tier; dropped for BM25
- `llama-3.1-70b-versatile` — decommissioned by Groq; replaced with `llama-3.3-70b-versatile`
- `groq==0.11.0` — incompatible with `httpx>=0.28` (`proxies` kwarg removed); upgraded to `groq>=1.2.0`
- LLM sometimes returned raw JSON instead of code-fenced block — added fallback regex
- LLM over-clarified even with sufficient context — added explicit recommend-vs-clarify rules and JD-paste detection
- `scraper.py` `html` parameter shadowed stdlib `html` module — renamed to `page_html`

**AI tools used:** Claude Code (Anthropic) for code generation and iteration. All design choices reviewed and understood; architecture, retrieval strategy, and prompt rules authored with full understanding of trade-offs.

---

## Stack Summary

| Component | Choice | Reason |
|---|---|---|
| Framework | FastAPI | Fast, schema validation via Pydantic |
| LLM | Groq / Llama-3.3-70b | Fastest free inference, fits 30s timeout |
| Retrieval | BM25 (rank-bm25) + boost + expansion | Fits 512MB RAM; role-context expansion closes recall gap vs. semantic search |
| Deployment | Render (Docker) | Free tier, cold start within 2-min health window |
