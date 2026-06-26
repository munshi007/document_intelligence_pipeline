# Publishing Pitch — talking points for the professor meeting

> Companion to [`SOTA_AUDIT.md`](SOTA_AUDIT.md) (the full evidence) and
> [`CALIBRATION.md`](CALIBRATION.md) (the one finished experiment). This is the
> *meeting script*: what to claim, what to concede, what's left. Honesty-first —
> the literature pass (SOTA_AUDIT §6) already found the obvious novelty claim is
> taken, so the credible pitch leads with that, not around it.

---

## 0. The 30-second pitch (say this first)

> "We built a **zero-training, model-agnostic, post-hoc grounding-and-repair
> layer** that sits on top of *any* schema-conditioned LLM extractor. Every value
> in the output JSON is verified back to a verbatim span in the source, repaired
> if it's a notation mismatch, flagged if it can't be grounded, and assigned a
> **calibrated confidence** — which we measured (ECE 0.165 → 0.03 after the
> normalization fix). The contribution is the **integration + the calibration
> evidence + the open implementation**, not a new algorithm. We cite the prior
> art for each component honestly."

That framing is defensible. The "we invented span-grounding" framing is **not** —
see §3.

---

## 1. What is actually publishable (lead with these)

| # | Contribution | Evidence on disk | Strength |
|---|---|---|---|
| **C1** | **A calibrated faithfulness number for grounding confidence.** Most extraction systems emit `confidence=1.0` constants (ours used to, in 8 places). We replaced that with a per-leaf `field_confidence` and then *measured its reliability* — ECE / MCE / Brier over a hand-labelled set. | `docs/CALIBRATION.md`, `docs/calibration_data/calibration_report.json`, `extractor/calibration.py` (+ unit tests) | **Strongest.** Concrete, reproducible, and rare — most papers never calibrate. |
| **C2** | **A diagnosed-and-fixed calibration failure.** The ECE table *root-caused* the miscalibration to narrow numeric/unit normalization (comma-decimals, ± glyphs, scale-words), and the fix (value-preserving notation folding) cut ECE 0.135 → 0.032 on identical source. This is a clean "measure → diagnose → fix → re-measure" loop. | `extractor/agent.py` (`_verify_string_spans`), `tests/extraction/test_grounding_normalization.py`, `scripts/validate_ece_delta.py` | Strong — shows the calibration is *actionable*, not decorative. |
| **C3** | **The integration delta:** a fully post-hoc grounding+repair layer over **arbitrary nested JSON schemas** (vs LangExtract's flat entity lists, AEVS's triples, Extract-0's fine-tuned model). Zero training, model-agnostic. | `extractor/agent.py` verify/retry stack; `--hint-fields` user-in-loop schema knob | Moderate — defensible *only* as a systems contribution that cites the components. |

**One-liner for each, if he asks "so what's new?":**
- C1: "We don't just emit a confidence — we proved what it's worth."
- C2: "We found *why* it was wrong and closed the gap, with before/after numbers."
- C3: "Nobody else does this over arbitrary nested schemas with zero training."

---

## 2. What's NOT done yet (be upfront — these gate submission)

| Gap | Status | Why it matters for a paper |
|---|---|---|
| **Vision (Killer #1)** | ⬜ not started | The extractor never sees pixels (`image=None`). We're text-only vs Qwen2.5-VL. Either wire it in (action **a**: route to vision when grounding confidence is low — the calibrated score *is* that routing signal) or scope the paper to "text+layout grounding" explicitly. |
| **End-to-end accuracy vs baselines** | ⬜ not started | No public-benchmark comparison vs Extract-0 / Qwen2.5-VL yet. A reviewer will want F1/edit-distance on FUNSD/CORD/OmniDocBench, not just our internal ECE. |
| **Single-domain calibration** | ⚠️ partial | All 24 calibration docs are Murr connector datasheets. Need ≥2 more domains (invoices, forms) before the ECE claim generalizes. |
| **Single-annotator labels** | ⚠️ partial | The 107 support-labels are author-judged. A second annotator (even on a subset, for κ agreement) hardens it. |
| **Venue + deadline** | ⬜ undecided | Not selected. See §5. |

Saying these *first* makes the professor trust the parts you claim are done.

---

## 3. The novelty reality — concede this proactively

The literature pass (SOTA_AUDIT §6, web-verified) found the grounding *constellation*
is **prior art**. Lead with this; it reads as rigor.

**Concede up front:**
- **Span verification + null-retry + auto-provenance as a bundle** → SciEx (arXiv:2512.10004), AEVS (MDPI Computers 2026).
- **`SequenceMatcher` fuzzy span grounding** → LangExtract (`resolver.py`, Google OSS).
- **Recursive schema-tree filling** → SPIRES (arXiv:2304.02711).

**Do NOT claim** any of those three as novel — a reviewer greps them in 5 minutes.

**What genuinely survives:**
- The **calibration evidence** (C1/C2) — the prior-art systems above don't publish an ECE table for their grounding confidence. *This is your wedge.*
- The **zero-training, model-agnostic, arbitrary-nested-schema** integration (C3) — defensible as engineering, not method invention.
- The **digit/notation-aware normalization** trick — no published grounding method names it, but it reads as engineering, so use it as supporting detail, not a headline.

---

## 4. Two honest paper framings (pick one with him)

**Option A — Systems / engineering paper (recommended, strongest survivor).**
> *"A zero-training, model-agnostic, post-hoc grounding-and-repair layer for
> arbitrary JSON-schema extraction — with calibrated faithfulness."*
- Contribution = integration + the calibration study + open implementation.
- Cites SciEx / LangExtract / AEVS / SPIRES as composed components.
- Venue fit: DocAI / NLP-systems workshops (ACL/EMNLP), or a tools/dataset track.
- **Lowest risk** — the work is mostly done; needs the multi-domain calibration + one baseline table.

**Option B — Benchmark / faithfulness paper (higher ceiling, higher risk).**
> Contribute a **grounding-faithfulness benchmark**: for each schema leaf, is it
> verbatim-supported? Are the auto-citations correct? Score systems on it.
- Differentiates from saturated OmniDocBench (and incoming ParseBench).
- **Risk:** benchmark papers need scale (many docs, many systems scored, inter-annotator agreement) — much more labelling than we have now.

Recommendation: **pitch A, mention B as a follow-up.** A is publishable from where we
are with ~3–4 weeks of work; B is a second paper.

---

## 5. Suggested venues + rough timeline

- **First move:** arXiv preprint (stake the claim, get feedback). No gatekeeping.
- **Workshop:** an ACL/EMNLP/NAACL **Document Intelligence / Structured-Extraction
  workshop** — right scope for a systems contribution, faster review than main track.
- **Step up if baselines are strong:** main-conference short paper or a journal
  (the calibration angle suits a reproducibility/measurement venue).
- **Rough plan (Option A):** ~1 wk multi-domain calibration → ~1 wk baseline table
  (Extract-0 / Qwen2.5-VL on a public set) → ~1 wk writing → arXiv → workshop deadline.

(Confirm actual deadlines with him — those drive everything.)

---

## 6. Anticipated reviewer/professor questions (rehearse these)

1. **"Isn't this just LangExtract / SciEx?"** → "The *components* are, and we cite
   them. Our contribution is the calibration evidence and the zero-training
   arbitrary-schema integration — neither of those papers reports an ECE table."
2. **"Why no vision? You call it a VLM pipeline."** → "Honest scoping: the current
   extractor is text+layout. The calibrated confidence is precisely the signal we'd
   route on to add vision (future work / Killer #1)." Don't oversell.
3. **"n=107, one domain, one annotator — is the ECE real?"** → "It's a first
   measurement with a committed audit trail; the binning-free Brier (0.131)
   corroborates it. Multi-domain + second-annotator is the next step (§2)."
4. **"What's the baseline accuracy?"** → Be honest if not done yet: "Internal
   grounding metrics are done; public-benchmark head-to-head is the immediate
   next experiment."
5. **"Calibration vs correctness?"** → "We measure *faithfulness* — is the value
   supported by the source — not factual correctness. We're explicit about that."

---

## 7. The one slide, if you make one

```
Contribution:  post-hoc, zero-training, model-agnostic grounding + repair
               over arbitrary nested JSON schemas — with CALIBRATED confidence.
Evidence:      ECE 0.165 (n=107, hand-labelled) → 0.03 after diagnosed fix.
Honest novelty: components are prior art (SciEx/LangExtract/AEVS/SPIRES, cited);
               the calibration study + integration are ours.
Next:          multi-domain calibration · public-benchmark baselines · (vision).
Framing:       systems paper now; faithfulness-benchmark paper as follow-up.
```
