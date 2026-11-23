# SafeDrive AI v4: Related Work and Positioning

This review was written **before** any claim of contribution. The final positioning (§4) was revised **after** the experiments
(§5 of [SAFEDRIVE_V4_EXPERIMENT_PLAN.md](SAFEDRIVE_V4_EXPERIMENT_PLAN.md)).

Sources were retrieved on 2026-11-23. Each was checked against its arXiv, publisher or official proceedings page; links are in
§6. Two classic references (Lee 1976; Chan et al. 2016) are cited from established knowledge rather than re-retrieved, and are
marked †.

## 1. Accident anticipation from dashcam video
| Work | Idea | Relevance to v4 |
|---|---|---|
| Chan et al., ACCV 2016† (DSA) | Dynamic soft attention over detected objects, RNN; DAD dataset | Object-centric anticipation from dashcam: the starting point of the field |
| Bao, Yu, Kong, ACM MM 2020 (UString, arXiv:2008.00334) | Graph convolution over agents on RNN hidden states, plus a Bayesian head for uncertainty; CCD dataset | Closest to direction B (graph + temporal model + uncertainty). Its relations are learned, not symbolic |
| Karim et al., IEEE T-ITS 23(7) 2022 (DSTA, arXiv:2106.10197) | Dynamic spatial and temporal attention, GRU | A strong learned baseline; no explicit relations |
| Thakur, Gouripeddi, Li, WACV 2024 (Graph(Graph)) | Nested graph of objects within and across frames, plus global features | Graph-based anticipation. Its authors stress that earlier methods raise many false positives |
| Moura, Zhu, Zvitia, CVPRW 2025 (arXiv:2503.03848) | Nexar dashcam collision-prediction dataset (1,500 clips), AP at 0.5 / 1.0 / 1.5 s | **The dataset and metric used by SafeDrive** (window mAP = the Nexar protocol) |

## 2. Scene graphs for driving risk
| Work | Idea | Relevance |
|---|---|---|
| Yu, Malawade et al., IEEE T-ITS 23(7) 2022 (arXiv:2009.06435) | Scene graphs, MR-GCN, LSTM and attention for subjective risk of lane changes | Scene graphs as an intermediate representation for risk; evaluated on synthetic and real lane-change data |
| Malawade et al., arXiv:2111.06123 (SG2VEC; roadscene2vec) | Spatio-temporal scene-graph embedding (MR-GCN + LSTM) for collision prediction | The closest prior art to "Dynamic Traffic Scene Graph → GNN → collision prediction" (direction B) |

## 3. Knowledge graphs and explainability for automated driving
| Work | Idea | Relevance |
|---|---|---|
| Luettin, Monka, Henson, Halilaj, KGSWC 2022 (arXiv:2210.08119) | Survey of ontologies and KGs for automated driving | Shows that KG-based scene understanding is an established direction |
| Halilaj et al., ESWC 2021 (CoSI) | KG fusing heterogeneous context and situation information | KG for situation understanding |
| Mlodzian et al., ICCVW 2023 (nuScenes Knowledge Graph, arXiv:2312.09676) | Ontology-based KG of agents, map and relations for trajectory prediction | Rich KG built from **annotated HD maps**. SafeDrive has only a monocular camera and no map |
| Atakishiyev et al. (arXiv:2112.11561; 2403.12176) and Kuznietsov et al. (arXiv:2402.10086) | Surveys of XAI for autonomous driving, and of its safety implications | Framing of explanation requirements and of their limits |
| Lundberg et al., Nature Machine Intelligence 2(1), 2020 (TreeExplainer, arXiv:1905.04610) | Exact polynomial-time Shapley values for tree ensembles | **Used as-is** (LightGBM's built-in TreeSHAP) |
| Jacob et al., ECCV 2022 (STEEX); Zemni et al. (OCTET) | Generative image counterfactuals for driving-scene classifiers | Pixel-level counterfactuals. v4 instead edits **trajectories** and recomputes features |
| Lee, Perception 1976† | Visual time-to-contact tau = s / (ds/dt) | Basis of the image-based TTC proxy (already used in v3) |

## 4. What v4 inherits, what differs, what is not claimed
**Inherited (not new):**
- scene graphs of road users with typed relations;
- graph neural networks over scene graphs combined with temporal recurrence;
- rule- and ontology-based traffic knowledge;
- TreeSHAP;
- counterfactual explanations;
- ensemble uncertainty.

**Differences of this implementation, stated as engineering choices, not as scientific novelty:**
1. **Monocular, map-free, ego-motion-compensated** scene graph built from a cheap dashcam pipeline (YOLO + ByteTrack + homography).
   Relations are emitted only with evidence and temporal persistence. A relation the image cannot support (heading, so
   OPPOSITE_DIRECTION) is disabled rather than guessed.
2. **Rules produce evidence, not warnings.** The KG rules are inputs to a learned, calibrated predictor and are evaluated by
   ablation on the Nexar protocol, with fixed false-alarm budgets.
3. **Every warning carries a sealed evidence trace** that links SHAP factors to graph nodes, edges, timestamps and rule
   findings, together with what the system could not determine.
4. **Trajectory-level counterfactuals** recompute all features from the edited tracks. Identity reproduction was verified to be
   exact.

**Not claimed:**
- "first", "novel framework", or any superiority over the methods above. None of them was re-run on Nexar here, so no
  head-to-head comparison exists;
- safety certification, or prevention of accidents.

The experiments (experiment plan §5) decide only one question: **whether v4 is better than v3 on the same protocol.**

## 5. Gaps that remain
- No comparison with UString, DSTA, Graph(Graph) or SG2VEC on Nexar. Their code exists for other datasets, but adapting them is
  a separate project.
- No human evaluation of explanations (the annotators are not trained yet).
- No real-world driving data (0.0 h).

## 6. Sources
- Bao, Yu, Kong: [arXiv:2008.00334](https://arxiv.org/abs/2008.00334v1), [DOI 10.1145/3394171.3413827](https://www.doi.org/10.1145/3394171.3413827)
- Karim et al.: [arXiv:2106.10197](https://arxiv.org/abs/2106.10197v2), [code](https://github.com/monjurulkarim/DSTA)
- Thakur et al.: [WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/papers/Thakur_GraphGraph_A_Nested_Graph-Based_Framework_for_Early_Accident_Anticipation_WACV_2024_paper.pdf)
- Moura et al.: [arXiv:2503.03848](https://arxiv.org/abs/2503.03848), [CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.html)
- Yu et al.: [arXiv:2009.06435](https://arxiv.org/abs/2009.06435)
- Malawade et al.: [arXiv:2111.06123](https://arxiv.org/abs/2111.06123v1)
- Luettin et al.: [arXiv:2210.08119](https://arxiv.org/abs/2210.08119v1)
- Halilaj et al. (CoSI): [Springer, DOI 10.1007/978-3-030-77385-4_42](https://link.springer.com/chapter/10.1007/978-3-030-77385-4_42)
- Mlodzian et al.: [arXiv:2312.09676](https://arxiv.org/pdf/2312.09676), [ICCVW 2023](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/papers/Mlodzian_nuScenes_Knowledge_Graph_-_A_Comprehensive_Semantic_Representation_of_Traffic_ICCVW_2023_paper.pdf)
- Atakishiyev et al.: [arXiv:2112.11561](https://arxiv.org/abs/2112.11561v3), [arXiv:2403.12176](https://arxiv.org/pdf/2403.12176)
- Kuznietsov et al.: [arXiv:2402.10086](https://arxiv.org/html/2402.10086v2)
- Lundberg et al.: [arXiv:1905.04610](https://arxiv.org/pdf/1905.04610)
- Jacob et al. (STEEX): [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4953_ECCV_2022_paper.php), [code](https://github.com/valeoai/STEEX)

## 7. Added for the V4 product round (uncertainty, selective prediction, faithful explanations)
Retrieved on 2026-11-23 and checked against publisher or arXiv pages.

| Work | Idea | Use in V4 |
|---|---|---|
| Jøsang, *Subjective Logic*, Springer 2016 ([DOI 10.1007/978-3-319-42337-1](https://link.springer.com/doi/10.1007/978-3-319-42337-1)) | Binomial opinions (belief, disbelief, uncertainty, base rate) ≡ Beta evidence; conjunction, cumulative fusion, discounting | **Basis of the uncertainty-aware KG** (edge / node / finding / temporal opinions) |
| Zhao, Chen, Hu, Cho, NeurIPS 2020 ([arXiv:2010.12783](https://arxiv.org/abs/2010.12783v2)) | Subjective-logic uncertainty (vacuity, dissonance) on graph data | Shows subjective logic as an uncertainty model on graphs; V4 uses it in symbolic reasoning, not in a GNN |
| Sensoy, Kaplan, Kandemir, NeurIPS 2018 ([arXiv:1806.01768](https://arxiv.org/abs/1806.01768v3)) | Evidential deep learning: Dirichlet / subjective-logic outputs | Related evidence-based uncertainty. Not used, because the v3 predictor is a frozen tree ensemble |
| Geifman, El-Yaniv, NeurIPS 2017 ([arXiv:1705.08500](https://arxiv.org/abs/1705.08500)) | Selective classification: risk–coverage trade-off with a reject option | Frame of H4 (coverage, selective risk of reliability-gated warnings) |
| Jacovi, Goldberg, ACL 2020 ([2020.acl-main.386](https://aclanthology.org/2020.acl-main.386)) | Faithfulness is not plausibility, and should be evaluated as a graded property | Why H3 separates **verifiability** (graph evidence) from **faithfulness to the model** |
| STAGNet (arXiv:2508.15216), depth-aware anticipation (arXiv:2502.18496) | Recent graph-attention / occlusion-aware accident anticipation with uncertainty | More recent related work. Not compared on Nexar |

**Position after the experiments.**
- **What V4 does differently.** It combines a monocular, map-free dynamic scene graph, a rule-based traffic KG with
  subjective-logic uncertainty, evidence-grounded sealed traces, and a 5-state warning policy in one product.
- **What it does not show.** The experiments do **not** show a predictive or robustness gain from this combination (H1, H2, H4
  not supported).
- **The only claim it supports** is the verifiability of its explanations (H3, partly).
- **Not claimed:** "novel", "first" or "state of the art".
