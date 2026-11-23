# SafeDrive AI v4: Traffic Knowledge Graph and Reasoning (kg-1.0)

Code:
- `safedrive_v4/knowledge_graph/ontology.py`
- `safedrive_v4/reasoning/rules.py`
- `safedrive_v4/reasoning/engine.py`

## 1. Representation choice
| Option | Assessment |
|---|---|
| RDF / OWL | Standard and interoperable, but needs rdflib plus a reasoner (not installed). v4 uses no description-logic inference, so the cost is not justified |
| Property graph (in-process) | Typed nodes and edges with attributes; matches the scene-graph data model |
| NetworkX | **Chosen.** Already installed (3.7), in-process, inspectable, exportable to node-link JSON and subject-predicate-object triples |
| Neo4j | A database server for a graph of about 70 nodes; no multi-user or persistence need. **Not used** |

Exports:
- `traffic_kg.nodelink.json`
- `traffic_kg.triples.tsv`

Both are written by `knowledge_graph.ontology.export`.

## 2. Ontology
Classes (SUBCLASS_OF hierarchy):
- **Road users.**
  - RoadUser → Vehicle → {EgoVehicle, Car, Bus, Truck, Motorcycle}.
  - RoadUser → {Bicycle, Pedestrian}.
- **Scenarios and interactions.**
  - RoadScenario → {CutInScenario, CarFollowingScenario, CrossingScenario}.
  - TrafficInteraction.
- **Risk factors.** RiskFactor → {LowTTC, HighClosingSpeed, LateralIntrusion, TrajectoryConflict, VulnerableRoadUserInPath,
  LeadDeceleration, Occlusion, PerceptionDegradation}.
- **Risk levels.** RiskLevel → {PotentialCollisionRisk, ElevatedRisk, LowRisk}.
- **Traffic rules.** TrafficRule → {KeepSafeDistance, YieldToPedestrians, SafeLaneChange}.
- **Other.** Evidence, Warning, and Uncertainty → {ObservationReliability, InsufficientEvidence}.

Each scene-graph relation INSTANTIATES an ontology concept. Example: `rel:CUTTING_IN_FRONT_OF` → CutInScenario.

**Domain triples are hypotheses** (status `hypothesis`) to be tested, not ground truth. Examples:
- `Motorcycle CAN_PERFORM CutInScenario`
- `LowTTC INDICATES PotentialCollisionRisk`
- `Occlusion REDUCES ObservationReliability`
- `HighClosingSpeed INCREASES PotentialCollisionRisk`
- `ObservationReliability CONDITIONS Warning` (design: low reliability gives "insufficient evidence", not "high risk")

## 3. Rules
Every rule has: an id, trigger, inputs, KG relations, base confidence, an explanation template (filled **only with measured
values**) and exceptions. Rules produce **findings**, never warnings.

| ID | Kind | Trigger | KG relations used | Base conf. | Exceptions |
|---|---|---|---|---|---|
| R01 cut-in | risk | CUTTING_IN_FRONT_OF valid | Vehicle CAN_PERFORM CutInScenario; LateralIntrusion INCREASES PotentialCollisionRisk | 0.80 | ego turning ×0.5; occluded / truncated ×0.7 |
| R02 fast approach | risk | APPROACHING, tau < 2.5 s | LowTTC INDICATES PotentialCollisionRisk | 0.70 | tau is image-based; occluded ×0.7; turning ×0.5 |
| R03 low TTC + high closing | risk | tau < 1.5 s and g > 0.5 /s | HighClosingSpeed INCREASES …; LowTTC INDICATES … | 0.85 | as R02 |
| R04 VRU crossing | risk | CROSSING_PATH (pedestrian / bicycle) | VulnerableRoadUserInPath INCREASES … | 0.75 | far road user ×0.5; occluded ×0.7 |
| R05 lead deceleration proxy | risk | FOLLOWING, lead growth rising > 1 /s per s | CarFollowingScenario CAN_INVOLVE LeadDeceleration | 0.60 | no brake-light detection; ego acceleration can mimic it |
| R06 trajectory conflict | risk | POTENTIAL_CONFLICT valid | TrajectoryConflict INDICATES … | 0.60 | constant image velocity assumed |
| R07 occlusion | reliability | OCCLUDING near the path | Occlusion REDUCES ObservationReliability | 0.60 | — |
| R08 perception degradation | reliability | dark frame, low detector confidence or unstable tracks | PerceptionDegradation REDUCES ObservationReliability | 0.80 | — |
| R09 insufficient evidence | insufficient | corridor objects without a reliable motion estimate | ObservationReliability CONDITIONS Warning | 0.90 | — |
| R10 interaction ahead | risk (weak) | INTERACTING_WITH between road users in the path regions | TrafficInteraction INCREASES ElevatedRisk | 0.40 | no ego involvement |

**Engine.**
- *Conflicts.* Risk evidence on a partially occluded object is reduced ×0.8. Low TTC on an object also judged SAME_DIRECTION
  (co-moving) is reduced ×0.5.
- *Summary scores.* risk_evidence is the noisy-OR of the risk findings and is **not** a probability. reliability is 1 −
  noisy-OR of the reliability findings. The `insufficient` flag is set when R09 fires.
- *Cost.* 0.045 ms per step.

**What the KG must not do, and does not:**
- infer unobserved facts (traffic lights, lane markings, distances);
- turn a rule firing into a warning.

Whether the rules carry signal is measured by ablation (A2, A5).

## 4. Evaluation of the rules
See the experiment plan, §5:
- configurations A2 (v3 + KG features) and A5 (KG features alone);
- the SHAP share of KG features;
- per-rule firing statistics on positive vs negative windows.

## 5. Uncertainty-aware reasoning (V4 product round)
**Opinions.** Every edge and node carries a subjective-logic opinion. The engine `reasoning/uncertain_engine.py` combines them
with:
- conjunction (premises);
- discounting (exceptions, rule trust);
- cumulative fusion (independent road users);
- temporal aging.

Details: [SAFEDRIVE_V4_UNCERTAINTY.md](SAFEDRIVE_V4_UNCERTAINTY.md).

**Role of the KG in the product (acceptance criterion 4).** The KG takes part in reasoning, explanations, the event category
(potential near-miss needs KG risk evidence) and the optional evidence gate. As a predictor input it was tested (A2, A3, A5) and
did not help (experiment plan §3–4).

**Rule statistics** (dev only, `experiments/v4/rule_stats_dev.json`):

| Rule | 2 s before events | normal driving | lift | onsets / min (normal) |
|---|---|---|---|---|
| R01 cut-in | 0.079 | 0.039 | 2.0 | 4.9 |
| R02 fast approach | 0.277 | 0.139 | 2.0 | 19.1 |
| R03 low TTC + closing | 0.155 | 0.081 | 1.9 | 14.0 |
| R04 VRU crossing | 0.001 | 0.003 | 0.2 | 0.3 |
| R05 lead deceleration | 0.069 | 0.025 | 2.7 | 11.9 |
| R06 trajectory conflict | 0.137 | 0.067 | 2.0 | 10.3 |
| R07 occlusion | 0.046 | 0.042 | 1.1 | 8.3 |
| R08 perception degradation | 0.212 | 0.298 | 0.7 | 24.3 |
| R09 insufficient evidence | 0.355 | 0.341 | 1.0 | 51.0 |
| R10 interaction ahead | 0.071 | 0.058 | 1.2 | 7.9 |

**Reading the table.**
- The risk rules carry signal: they fire about 2× more often just before events.
- They are far too frequent to serve as alarms: 5–19 times per minute, against an alarm budget of 5–20 per hour.
- The hypothesis "VulnerableRoadUserInPath INCREASES risk" is **untested** on Nexar: R04 almost never fires.
