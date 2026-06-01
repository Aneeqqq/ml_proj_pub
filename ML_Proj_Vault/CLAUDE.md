---
name: vault-schema
description: Schema and operating manual for the ML_Proj knowledge vault (how the wiki is structured and maintained)
metadata:
  type: reference
---

# ML_Proj_Vault — Schema & Operating Manual

This is an **LLM-maintained wiki** (the "LLM Wiki" / Memex pattern) for replicating the paper
**"Multi-Modal Sensor Fusion for Proactive Blockage Prediction in mmWave Vehicular Networks"**
(Nazar et al., arXiv:2507.15769v1, 2025) on the **DeepSense 6G — Scenario 31** dataset.

The vault is the **permanent memory** for this project. The LLM writes and maintains it; the human
curates sources, asks questions, and directs the work. Read [[00_overview]] first.

## Three layers (Memex / LLM-Wiki pattern)

1. **Raw sources** (immutable — never edited): live OUTSIDE the vault, at the repo root.
   - `../RLC_Bockage.pdf` — the paper (5 pages). Source of truth for architecture.
   - `../_pdf_text.txt`, `../_fig1.png`, `../_fig2.png` — extracted text & rendered diagrams (derived, disposable).
   - `../scenario31_new/scenario31/` — the dataset (1 scenario provided so far).
2. **The wiki** (this directory) — LLM-generated, interlinked markdown. You read it; the LLM writes it.
3. **The schema** (this file) — conventions + workflows. Co-evolves with the project.

## Directory map

```
ML_Proj_Vault/
  CLAUDE.md                 <- this schema
  index.md                  <- content catalog (every page, 1-line summary)
  log.md                    <- chronological append-only event log
  00_overview.md            <- paper thesis + project goal + entry point
  paper/                    <- faithful capture of the paper
    paper-summary.md, system-model.md, architecture.md,
    problem-formulation.md, results.md
  modalities/               <- per-modality deep dives (camera & radar = priority)
    camera.md, radar.md, gps.md, lidar.md, fusion.md
  dataset/                  <- THE critical layer; get this right or everything downstream breaks
    scenario31-structure.md, sequences-and-batching.md,
    blockage-label.md, abnormalities.md
  concepts/                 <- reusable concept pages
    blockage-prediction.md, late-fusion.md, lstm-temporal.md, class-imbalance.md
  plan/
    replication-plan.md, open-questions.md
  lessons/
    lessons-learned.md
```

## Page conventions

- **YAML frontmatter** on every page: `title`, `tags`, `updated` (YYYY-MM-DD), `status`
  (`draft` | `verified` | `needs-source`), and where relevant `confidence` (paper-stated vs inferred).
- **Wikilinks** `[[page-name]]` (no `.md`, no path) to interlink liberally. Obsidian resolves them.
- **Distinguish fact provenance** explicitly:
  - `📄 PAPER` — stated in the paper (cite section, e.g. "§III-E").
  - `📊 DATA` — measured from the dataset (cite the analysis).
  - `⚠️ ABNORMALITY` — inconsistency / ambiguity / risk. Mirror into [[abnormalities]].
  - `❓ OPEN` — unresolved question. Mirror into [[open-questions]].
  - `💡 LESSON` — something learned the hard way. Mirror into [[lessons-learned]].
- Never silently "fix" a contradiction between paper and data — flag it as `⚠️ ABNORMALITY`.

## Workflows

### Ingest a new source (paper, dataset scenario, code)
1. Read the raw source fully (for PDFs: extract text AND render figures — LLMs miss diagrams otherwise).
2. Discuss key takeaways with the human.
3. Create/update the relevant wiki pages (a single source may touch 10–15 pages).
4. Update [[index]] (catalog) and append to [[log]] with prefix `## [YYYY-MM-DD] ingest | <title>`.
5. Surface any new `⚠️`/`❓`/`💡` into their hub pages.

### Query the wiki
1. Read [[index]] first to locate relevant pages, then drill in.
2. Synthesize with citations to pages/sections.
3. **File good answers back** as new pages (analyses, comparisons) — don't let them vanish into chat.

### Lint (periodic health check)
Look for: contradictions between pages, stale claims, orphan pages (no inbound links), concepts
mentioned but lacking a page, missing cross-references, data gaps fillable by analysis/web search.
Append a `## [YYYY-MM-DD] lint` entry to [[log]].

## Priority directive (from the human)

- **Stay faithful to the paper's architecture.** When in doubt, the PDF wins; record deviations.
- **Camera and radar are the focus** — keep [[camera]] and [[radar]] crystal clear and complete.
- **The dataset is make-or-break.** The data is organized into **scenes/sequences** (`seq_index`).
  Mishandling sequence boundaries or the windowing corrupts everything downstream. See
  [[sequences-and-batching]] before writing any dataloader.
