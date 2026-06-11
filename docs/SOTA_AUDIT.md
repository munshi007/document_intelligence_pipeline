# SOTA Audit, Novelty Analysis & Publishing Roadmap

> Consolidated output of the research tasks #42 (deep codebase audit), #43 (SOTA
> literature review), and #44 (novelty + componentization synthesis).
>
> **Provenance:** distilled from the working session; recovered from transcript
> and written to disk so it survives context compaction. Code line references
> were verified against the tree at the time of writing — re-check after large
> refactors.

## ⚠️ Read this first — what is and isn't confirmed

| Part | Status |
|---|---|
| **Codebase audit** (findings, line numbers) | ✅ Code-verified |
| **SOTA landscape / benchmark numbers** | ✅ **Web-verified 2026-06-08** (see §3 + §6). One correction: OCRBench = 885/1000, not "88.8". |
| **"Is the post-hoc grounding angle already published?"** | ✅ **RESOLVED 2026-06-08 — and the answer is bad: largely YES.** Headline methods-novelty is **WEAKENED→DEAD** (SciEx, LangExtract, AEVS, SPIRES are prior art). See §6. |
| **Paper-publishing roadmap** | ⚠️ **Skeleton only**, and now needs reframing to an engineering/systems or benchmark contribution (the methods-novelty headline did not survive). |

> **2026-06-08 UPDATE — the literature pass changed the conclusion.** The original
> audit assumed the grounding constellation was probably novel. A web-verified
> prior-art pass found it is **not**. See the new **§6 — Prior-Art Verification**
> for the full verdict and citations. Read §6 before acting on §5.

---

## TL;DR (three sentences)

1. The paper's headline contribution is the **source-grounded verify/retry stack** in
   `extractor/agent.py:306-1255` (multi-strategy span verifier + schema-tree-driven
   null-leaf retry + auto-provenance); closest published work found is span-level
   hallucination detection for *QA* (arXiv 2504.18639) — nobody appears to have done
   this exact constellation for *schema-conditioned structured extraction*.
2. **Three killers must be fixed before submission:** (1) the extractor never sees
   pixels (`image=None` everywhere), (2) `confidence_score=1.0` constant in 8 places,
   (3) hardcoded industrial vocab (`Murrelektronik`/`Pepperl+Fuchs`/`art_no`) in the
   production codepath — which also violates the project's own no-hardcoding rule.
3. **Componentize first:** the grounding verifier, the table router, and the HKG
   chunker are each a `pip install` waiting to happen, with industry pull independent
   of the paper.

---

## 1. Codebase audit — verified findings

### 🟢 Actually novel (paper-worthy)

| # | What | Where | Why novel |
|---|---|---|---|
| 1 | **Source-grounded post-hoc retry with verbatim-span enforcement** | `extractor/agent.py:306-1255` | Multi-strategy verifier (verbatim → SequenceMatcher windowed fuzzy → digit-aware fallback for `"2020/4/6" ↔ "20046"`), schema-tree retry (drills into wholly-null sub-objects), anchored source windows. The digit-snap is genuinely clever — not seen in LangExtract, Marker, Docling, or instructor. |
| 2 | **Schema-shape cross-validation between regex heuristic and LLM** | `discovery_agent.py:401-480` (`_DOMAIN_TELLTALES`) | Two-axis check: label × schema-shape. Catches the documented "Qwen industrial-bias" failure mode where the LLM agrees with the domain label but emits a schema shape from a different domain. Citeable. |
| 3 | **`--hint-fields` user-in-the-loop schema augmentation** | `stages/discovery.py:30-71` | Small but real — companies want this knob without writing JSON Schema. Related work (LLMs4SchemaDiscovery, ScheMatiQ, both 2025) targets *scientific corpora* — different flavor. |

**Not novel:** the HKG (`chunker/graph_builder.py`) — atomic chunking + caption
proximity + regex cross-refs is the Docling/Marker playbook. Solid engineering,
supporting chapter, not the headline.

### 🔴 Critical weaknesses (must-fix before submission — "the three killers")

| # | Weakness | Code evidence | Severity |
|---|---|---|---|
| 1 | **Visual context loss is total** | Every extractor call site passes `image=None`: `stages/extract.py:240`, `agent.py:1219/1297/1774/1891/1947`. Comment at 1891 literally says *"disable the image to prevent visual hallucinations"*. | **Catastrophic for a VLM-extractor claim** — text-only competing with Qwen2.5-VL (96.4% DocVQA) at a 5–15 F1 disadvantage on FUNSD/CORD. The HKG is a *substitute* for vision, not equivalent. |
| 2 | **`confidence_score=1.0` constant in 8 places** | `schema_definitions.py:136/178/226/237/267/315`, `discovery_agent.py:825/962`, `agent.py:1973`, plus `SourceEvidence.confidence=1.0` | A reviewer will hammer this — ECE/Brier reliability has been an open subfield for 2 years. The grounding audit already contains enough signal to compute a real calibrated number; nobody wired it up. |
| 3 | **Hardcoded industrial vocab in the "agnostic" path** | `agent.py:514-523` (manufacturer string checks — **PARTLY FIXED**, see status below); `agent.py:194` (prompt examples `art_no, parameters, connectors`); `agent.py:1698-1744` (parameters-dedup hardcoded to field `parameters`); `agent.py:1804-1810` (grounding module whitelist `electrical, mechanical, connectors, diagnostics, invoice_header`); `discovery_agent.py:548-567` (scout few-shot shows `art_no`); `schema_engine.py:251` | Any reviewer greps vendor names in 5 min and finds these. Violates the project's own rule. |

### 🟡 Other major weaknesses

4. **LayoutLMv3 is an un-fine-tuned base model** (`modules/layoutlm_classifier.py:42-58`).
   Classification head randomly initialized; comment at line ~186 admits *"for
   production, fine-tune on your dataset."* The rule-based fallback
   `_classify_rule_based:258-284` does the actual work. README implies a working
   "LayoutLMv3 classifier" → citation-claim risk.
5. **No error-cascade recovery.** Bad YOLO → bad markdown → bad extract, no rollback,
   no per-region confidence propagated. The only "fallback" at `agent.py:1437-1450`
   swaps to `LibrarianUniversalHardware` on zero-batches — i.e. industrial-bias
   recovery, not a generic mechanism.
6. **Tests are plumbing-only.** ~4 files, ~136 lines, 3/4 tautological (e.g.
   `tests/extraction/test_invoice_schema_path.py` asserts
   `get_schema_family("Corporate") == "invoice_v1"`). No grounding test, no HKG test,
   no end-to-end accuracy test. Likely the single biggest reviewer complaint.
7. **Throughput: single-doc, serial-page, serial-batch, no glob.** `cli.py` takes one
   PDF; `pipeline/enhanced_pipeline.py:281` iterates pages serially; `agent.py:1355`
   iterates batches serially. Poor GPU utilization.

---

## 2. Five prioritized code upgrades

| # | Change | File(s) | Buys you |
|---|---|---|---|
| **a** | Replace `image=None` with per-page images; route to vision provider when grounding pass-rate < threshold | `stages/extract.py:240` (plumbing exists at `agent.py:1297`) | +3–8 F1 on FUNSD/CORD |
| **b** | Wire `_last_grounding_stats` into per-field **calibrated** confidence; replace all `Field(1.0)` with computed values | `agent.py:1607` exists; defaults in `discovery_agent.py:825,962` + `schema_definitions.py` | Publishable ECE table for free + industry credibility |
| **c** | Purge industrial vocab from agnostic paths | `agent.py:194,514-523,1698-1744,1804-1810`; `discovery_agent.py:279,401-419,548-567`; `schema_engine.py:251` | Correctness + fixes the rule violation |
| **d** | Fine-tune LayoutLMv3 on existing `data/ground_truth/` for heading-level + region typing | `modules/layoutlm_classifier.py` | Publishable chapter: "learned multi-modal HKG construction" |
| **e** | Replace fixed `max_retries=5` with budget = `count(flagged) + 0.3 × count(null_leaves)`, early-stop on convergence | `agent.py:1553` | Tight latency/accuracy curve to publish |

---

## 3. SOTA landscape (as of ~May 2026 — NOT live-web re-verified)

### The bar to beat

| System | DocVQA | OCRBench | OmniDocBench (EN edit dist) | Year |
|---|---|---|---|---|
| **Qwen2.5-VL-72B** | **96.4%** (near-human 98.1%) | **88.8%** | **0.226** (lead) | Feb 2025 (arXiv 2502.13923) |
| Qwen2.5-VL-7B (consumer hardware) | 95.7% | — | — | Feb 2025 |
| Claude 3.5 Sonnet | 95.2% | — | — | — |
| GPT-4o | comparable | — | — | — |
| mPLUG-DocOwl2 (8B, OCR-free, 324 tok/page) | "SOTA multi-page" | — | — | ACL 2025 (arXiv 2409.03420) |
| LayoutLMv3 BASE (FUNSD) | — | — | F1 = 90.29% | older |
| LayoutLMv2 (CORD / SROIE) | — | — | F1 = 96.01% / 97.81% | older |

OmniDocBench is reportedly **"saturated"** (LlamaIndex, April 2026 blog) — the
community is hunting for the next benchmark.

### Closest analogue — Extract-0

**Extract-0** (arXiv 2509.22906): a specialized **distilled model for document
information extraction**. Beats GPT-4.1 (0.573 vs 0.457 mean reward on 1000 doc
extraction tasks). **Trained for $196.** Closest published thing to this project —
confirms the distilled-specialist angle works, but means real differentiation is
required. **The grounding stack is that differentiation.**

### Open-source production landscape (2026)

- **Marker** — safest default; structure fidelity + image/table handling; restrictive
  license.
- (Docling, LangExtract, instructor, etc. — referenced as comparison baselines; the
  HKG approach mirrors the Docling/Marker chunking playbook.)

### The open literature question (BLOCKER)

Training-data memory does **not** recall a specific paper doing this exact
constellation — multi-strategy span verification + schema-tree retry + auto-provenance
for schema-conditioned extraction. **This must be confirmed against live literature
before any submission.** If someone already published it, the headline novelty
collapses and the project pivots to the componentization track.

---

## 4. Componentization plan

**Ship today as standalone (minimal cleanup):**
- `chunker/graph_builder.py` (~447 lines, pure Python) — competes with LangChain's
  `MarkdownHeaderTextSplitter` + Docling chunker.
- `extractor/evaluation.py` (~316 lines, no GPU) — `pip install doc-extract-eval`;
  grounding-aware metrics are rare.
- `processors/tables_v2/router.py` — ordered-threshold ruled/KV/complex classifier
  with TSR/VLM fallback.

**Refactor then ship — the headline asset:**
- Pull `_verify_string_spans`, `_retry_flagged_and_null_strings`, `_anchored_source`,
  `_walk_schema_leaf_strings` out of `agent.py` into a free-standing
  `extractor.grounding` module. **This is the novel asset and it is currently
  entangled with industrial heuristics — which is exactly why the vocab purge
  matters.**

**Can't untangle (deprecate):**
- `discovery_agent.py` heuristic skeletons (six hardcoded domains).
- `schema_engine.py` legacy — `DiscoveryAgent.scout` largely supersedes it.

---

## 5. Publishing roadmap (SKELETON — needs finishing)

> This section is intentionally thin. It was never completed because the literature
> blocker (§3) was unresolved. Do not treat as a finished plan.

**Proposed framing:** *"Constrained post-hoc grounding for hallucination suppression
in schema-conditioned LLM extraction."*

**Sketch of phases:**
1. **Stop-the-bleeding (no novelty, ~1 week):** fix killers #2 (`confidence=1.0`) and
   #3 (hardcoded vocab) — credibility prerequisites.
2. **Literature confirmation (BLOCKER):** web-verify that the grounding constellation
   is unpublished. Gate everything on this.
3. **Vision (killer #1):** wire pixels through the extractor; measure FUNSD/CORD delta.
4. **Calibration (upgrade b):** turn the grounding audit into a real ECE table.
5. **Experiments:** end-to-end accuracy on a public benchmark vs Extract-0 / Qwen2.5-VL
   baselines; ablation of the verifier strategies (verbatim vs +fuzzy vs +digit-snap).
6. **Venue:** TBD — not selected. Candidates implied by the related work are NLP/DocAI
   workshops (ACL/EMNLP) and arXiv preprint first.

**Still TODO for a real roadmap:** target venue + deadline, dataset licensing, compute
budget, baseline reproduction, and the literature confirmation above.

---

## 6. Prior-Art Verification (web-verified 2026-06-08)

> Result of a dedicated web search pass. **Verdict: methods-novelty WEAKENED →
> bordering DEAD for the headline framing.** The grounding *constellation* is not
> novel; the surviving paper is an engineering/systems or benchmark contribution.

### The threatening prior art (ranked)

1. **SciEx** — Li et al., *Exploring LLMs for Scientific Information Extraction Using
   the SciEx Framework*, arXiv:2512.10004 (Dec 2025). Schema-conditioned extraction
   + closed-loop where *"missing, uncertain, or low-confidence fields trigger targeted
   follow-up queries"* until convergence (= our schema-tree null-retry) + auto-provenance
   metadata on every element (= our auto-provenance). **Covers techniques #2 and #3
   together, in the same problem space.**
2. **LangExtract** (Google, OSS, `resolver.py`) — already ships exact substring +
   **`difflib.SequenceMatcher`/LCS fuzzy** span grounding with hallucination filtering
   of ungrounded extractions. **This is functionally our verifier (#1a + #1b) plus
   auto-provenance (#3).**
3. **AEVS** — *Grounded KG Extraction via LLMs: An Anchor-Constrained Framework with
   Provenance Tracking*, MDPI Computers 15(3):178 (2026). Exact→fuzzy→schema matching
   hierarchy + "Supplement" re-extraction rounds + provenance. (#1+#2+#3 for KG/triples.)
4. **SPIRES** — Caufield et al., arXiv:2304.02711 / Bioinformatics 2024. Recursive
   nested-schema-tree filling (= our #2).

### What survives

- **The "digit-aware span normalization" fallback (#1c)** — *no published grounding
  method names this trick*. BUT it's a standard data-cleaning primitive (strip
  separators, compare digit runs), so it reads as engineering, not research novelty.
- **The packaging:** a *zero-training, model-agnostic, fully post-hoc* grounding +
  null-repair layer over **arbitrary nested JSON schemas** (vs LangExtract's flat
  entity lists, AEVS's triples, Extract-0's fine-tuned model). This integration-level
  delta is defensible **only if framed as a systems contribution** that explicitly
  cites SciEx / LangExtract / AEVS / SPIRES as the composed components.

### Honest framing options (pick one)

1. **Systems/engineering paper (strongest survivor):** "A zero-training, model-agnostic,
   post-hoc grounding-and-repair layer for arbitrary JSON-schema extraction." Contribution
   is integration + open implementation, not method invention.
2. **Benchmark paper (riskier — OmniDocBench is saturated and ParseBench is incoming):**
   contribute a grounding/faithfulness benchmark scoring whether each schema leaf is
   verbatim-supported and whether auto-citations are correct.

### Do NOT claim
- that span-verification + null-retry + auto-provenance is a novel constellation (SciEx, AEVS);
- that SequenceMatcher fuzzy span grounding is new (LangExtract `resolver.py`);
- that schema-tree recursive filling is new (SPIRES).

### Corrected SOTA numbers
- Qwen2.5-VL-72B: DocVQA **96.4** ✓, OmniDocBench EN edit-dist **0.226** ✓,
  **OCRBench = 885/1000** (the earlier "88.8" was wrong) — arXiv:2502.13923.
- Extract-0: reward **0.573** vs GPT-4.1 0.457, trained **$196** ✓ (does NOT do
  grounding/provenance — relevant to the SOTA table only) — arXiv:2509.22906.
- "OmniDocBench saturated" — LlamaIndex blog, **published 24 Feb 2026** (not April);
  top models ~94–95%, GLM-OCR 94.6%.
- New since last audit: **ParseBench** (arXiv:2604.08538, proposed successor benchmark);
  **Qianfan-OCR** unified doc model (arXiv:2603.13398). Frontier now led by
  Gemini 3 / GPT-5.2 / Kimi 5.2-class VLMs.

---

## Progress against this audit

- ✅ **Killer #3 (partial):** `project_to_schema` (`agent.py:~450-530`, the old
  `514-523` vendor-string block) rewritten structural-only on branch
  `feat/schema-hints` (merged from `fix/purge-hardcoded-vocab`). Validated by 10-doc
  A/B: 684 leaves identical, 0 lost, grounding flat.
- ✅ **Killer #3 (prompt + dedup):** `agent.py:194` prompt examples de-hardcoded and
  the parameters-dedup (`1698-1744`) made structural (commit `08098f7`).
- ✅ **Killer #3 (grounding whitelist):** the `_ground_block` langextract stub
  (discarded LLM call + fabricated `context[:200]` evidence at page 1/conf 0.95)
  replaced by deterministic block grounding via the span verifier — real snippet,
  real page, ratio as confidence; the block whitelist
  (`electrical/.../invoice_header`) is gone: any sub-model declaring
  `source_evidence` is grounded, schema-agnostically.
- ✅ **Killer #3 (scout few-shot):** the hardware-datasheet few-shot
  (`art_no`/`manufacturer`/`technical_parameters` — structurally the expected
  discovery output for the evaluation datasheets, i.e. few-shot leakage on the
  eval domain) replaced with a structurally identical LAB TEST REPORT example
  (non-corpus domain). The structural lesson (nested identity object +
  `{name,value,unit}` measurement array) is preserved.
- ✅ **Killer #3 (schema_engine):** `build_runtime_schema_draft` (the
  keyword-triggered hardcoded schema injection at old line ~251) deleted —
  it had zero callers.
- **Killer #3 — remaining vocab is declared, not hidden:** the per-domain
  fallback skeletons (`discovery_agent.py:~279`) and `_DOMAIN_TELLTALES`
  (`~401`) are explicit domain-keyed registries: the former are documented
  domain priors, the latter a *defensive* vocabulary used to detect
  cross-domain hallucination in discovered skeletons. Neither masquerades as
  domain-agnostic logic; both are intentionally retained. Same for the
  regex signal lists in `schema_engine.py`'s heuristic domain/subtype router.
- ⬜ **Killer #1 (vision), #2 (calibration):** not started.
- ⬜ **Literature confirmation:** not started — **the gating blocker.**
