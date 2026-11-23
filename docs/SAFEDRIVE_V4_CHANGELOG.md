# SafeDrive AI v4.0.0-research: Changelog (2026-11-23)

**The v3 system is unchanged.**
- no file in `src/`, `pages/`, `ui/`, `models/safedrive_v3*` or `configs/` was modified;
- the v3 software hash stays 2032bf7b…;
- the frozen hashes and the v3 benchmarks are reproduced exactly (test report §2).

## New: `safedrive_v4/` (independent package)
| Path | Content |
|---|---|
| `configs/v4_default.json`, `config.py` | versioned config (`config_version`, hash recorded in artefacts) |
| `scene_graph/` | schema sg-1.0, incremental builder (12 relations with evidence and temporal validity) |
| `knowledge_graph/ontology.py` | ontology and domain triples (hypotheses), networkx property graph, triple / node-link export |
| `reasoning/` | 10 rules (id, trigger, inputs, KG relations, confidence, explanation, exceptions), engine with conflicts |
| `features/graph_features.py` | 36 scene-graph + 15 KG per-step channels; v3-identical window statistics |
| `prediction/` | graph tensors and temporal R-GCN + GRU (direction B) |
| `explainability/` | exact Tree SHAP (+ group level, stability, fidelity), graph reasoning chains, trajectory counterfactuals |
| `uncertainty/`, `warning/` | perception reliability, OOD, warning policy with INSUFFICIENT_EVIDENCE |
| `event_memory/trace.py` | sealed evidence traces |
| `pipeline.py` | streaming pipeline; `V3ReadOnlyModel` (A3) and `ResearchModel` (A6) |
| `dashboard/app_v4.py` | independent Streamlit app (port 8502) |
| `evaluation/` | precompute, ablation (v3 protocol), rule statistics, SHAP global, XAI evaluation |

## New artefacts
| Path | Content |
|---|---|
| `data/processed/v4_graph/sg-1.0_8f82a62f/` | per-video SG / KG channels and GNN tensors (≈ 60 MB) |
| `models/safedrive_v4_research/` | A6 research model (not used by the product) |
| `experiments/v4/` | ablation runs, `gnn_oof.json`, `rule_stats_dev.json`, `shap_global.json`, `xai_eval_a3.json`, logs, traces |

## Tests and docs
- **Tests:** `tests/test_v4_*.py`, 25 tests. Full suite: 189 passed.
- **Docs:** `docs/SAFEDRIVE_V4_{ARCHITECTURE, SCENE_GRAPH, KNOWLEDGE_GRAPH, XAI_DESIGN, EXPERIMENT_PLAN, RELATED_WORK, MODEL_CARD,
  TEST_REPORT, CHANGELOG}.md`.

## Findings recorded with this version
- No v4 feature set, alternative model or GNN improves on v3. The best held-out variant (A2) is ΔmAP −0.009, 95 % CI
  [−0.029, 0.007].
- **v4 is integrated as an explanation layer over the unchanged v3 predictor (A3).**
- Single-feature SHAP rankings are unstable across ensemble members, so explanations are shown at group level first.

# SafeDrive AI V4: product round (2026-11-23)
## New
| Path | Content |
|---|---|
| `uncertainty/subjective_logic.py`, `reasoning/uncertain_engine.py` | uncertainty-aware KG (subjective logic) |
| `uncertainty/input_quality.py` | DEGRADED_INPUT |
| `warning/policy.py` | 5 states, cautious HIGH, optional evidence gate (off) |
| `event_memory/store.py` | Near-Miss Intelligence |
| `app/` (`Home.py` + 7 pages, `core.py`, `live.py`, `ui.py`) | the single product app |
| `evaluation/precompute_u.py` | KGU channels and the perception-error model |
| `evaluation/hypotheses.py` | A3 / A5, H2, H4, with the v3 reproduction guard |

## Changed
| Path | Change |
|---|---|
| `scene_graph/builder.py` | edges carry a subjective-logic opinion and `uncertainty`; nodes carry a reliability opinion. Emitted edges are **unchanged**: identical SG / KG channels on 8 checked videos, so round-1 results stay valid |
| `pipeline.py` | per-step `step()` shared by replay, video analysis and live; uncertainty-aware engine; input quality; kgu buffer |

## Removed
- `safedrive_v4/dashboard/` and its test: replaced by the single app.

## Docs
- New: `SAFEDRIVE_V4_PRODUCT_SPEC.md`, `SAFEDRIVE_V4_UNCERTAINTY.md`.
- `XAI_DESIGN` renamed to `SAFEDRIVE_V4_XAI.md`.
- Updated: experiment plan (H1–H4 verdicts), related work, KG, architecture, model card, test report.

## Bug found and fixed (no result affected)
- `hypotheses.run2` first created a new random generator per training draw, so A0 did not reproduce v3. The reproduction guard
  now stops the run in that case; the invalid run was discarded.

## v3
- Unchanged: software hash 2032bf7b…, frozen hashes, held-out numbers reproduced.
