# SafeDrive AI v4: Dynamic Traffic Scene Graph (schema sg-1.0)

Code:
- `safedrive_v4/scene_graph/schema.py` (types)
- `safedrive_v4/scene_graph/builder.py` (incremental builder)

Tests: `tests/test_v4_graph_reasoning.py`.

## 1. Input and coordinate system
- **Input.** At each step (10 Hz), the tracked detections of the v3 perception: YOLO, then ByteTrack, giving
  `[track_id, coco_class, conf, x1, y1, x2, y2]` in normalised frame coordinates.
- **Ego motion.** The camera transform of that step, a homography estimated by v3 (`src/features/ego_motion.py`), plus scene
  context: optical-flow magnitude, median horizontal flow, brightness and mean detector confidence.
- **Image-plane only.**
  - Velocities are *scene-relative* image motion: the box at t−k is mapped through the accumulated camera transform, and the
    difference to the observed box is the object's own motion.
  - The scale growth `g` is in 1/s.
  - `tau_s = 1 / g` is the image time-to-contact proxy (Lee 1976), valid only under constant closing speed.
- **Not claimed.** No physical distance, speed or TTC is claimed. Unknown values stay `None`: a track younger than k = 3 steps
  has no velocity.

## 2. Nodes
| Type | Produced? | Attributes |
|---|---|---|
| EgoVehicle | always (`ego`) | forward-motion proxy (flow magnitude), turn proxy (median horizontal flow); image-based |
| Car, Bus, Truck, Motorcycle, Bicycle, Pedestrian | from tracks | object_id, class, confidence, bbox, centre, velocity, scale growth, acceleration, tau, bounded track history (≤ 20 steps), track age, visibility (visible / partially_occluded / truncated / new), uncertainty 0..1, region |
| RoadRegion | always: `ego_corridor` (\|x − 0.5\| < 0.2, y > 0.35), `left_adjacent`, `right_adjacent` | **fixed image regions**, no lane detection |
| TrafficLight | **never**: the v3 detector keeps only road users | schema slot only |
| RiskEvent | created by the pipeline for a warning (evidence trace), not by the builder | — |

**Node uncertainty** = 1 − reliability, where reliability = confidence term × track-age term × velocity stability ×
(0.8 if truncated). Partial occlusion adds 0.2.

## 3. Edges
Each edge has:
- source, target, relation type and confidence;
- timestamp;
- `evidence_features`, the numbers that triggered it;
- `temporal_validity`: first_seen, last_seen, valid_since and support (m of n).

**Emission rules:**
- both tracks are older than k steps (a velocity exists);
- confidence ≥ 0.30 and area ≥ 0.002;
- the condition holds in **≥ 2 of the last 3 steps, including the current one**.

| Relation | Condition (image-based) | Evidence |
|---|---|---|
| APPROACHING (obj→ego) | in corridor / adjacent, g > 0.15 /s | scale_growth, tau_s |
| MOVING_AWAY (obj→ego) | in corridor / adjacent, g < −0.15 /s | scale_growth |
| CUTTING_IN_FRONT_OF (obj→ego) | vehicle or motorcycle in an adjacent region, moving inward > 0.08 frame/s, reaching the corridor within 1.5 s | lateral_in, gap, time_to_corridor |
| CROSSING_PATH (obj→ego) | pedestrian or bicycle in the path region, lateral speed > 0.10 frame/s | lateral speed |
| NEAR (obj→ego) | bottom edge > 0.85 in corridor / adjacent | bottom_y, area |
| NEAR (obj↔obj) | centre distance < 1.5 × mean box size | relative distance |
| FOLLOWING (ego→lead) | lead = lowest box in the corridor, little lateral motion | lateral_in, bottom_y |
| SAME_DIRECTION (lead→ego) | ego moving (flow > 0.01) and lead scale nearly constant | g, ego flow |
| OPPOSITE_DIRECTION | **disabled by default** (`enable_opposite_direction=false`): a monocular image without lane or heading cues cannot support it | — |
| POTENTIAL_CONFLICT (obj→ego) | constant-velocity image prediction reaches the ego zone within 1.5 s, or tau < 2 s in the corridor | predicted point, tau |
| INTERACTING_WITH (obj↔obj) | converging pair, image time to meet < 2 s | closing rate, time to meet |
| OCCLUDING (front→back) | IoU > 0.2; front = lower bottom edge | IoU |
| LOCATED_IN (obj→region) | region membership | centre |

**Confidence** = min(node reliabilities) × (support / n) × evidence strength, clipped to 0..1.

## 4. Temporal graph
| Mechanism | Implementation |
|---|---|
| Sliding temporal window | window features use the steps in (t − W, t], with W ∈ {0.5, 1, 2} s compared in the ablation |
| Graph snapshot sequence | one `Snapshot` per step, consumed by the reasoning engine and by the GNN (per-step tensors) |
| Temporal edge update | a persistence deque per (source, target, relation); a key evaluated false is appended a miss, and keys are dropped when all-false |
| Track-level interaction history | per-track deques (positions, velocities, scale growth) of 20 steps; lost tracks are forgotten after 20 steps |

**Incremental and bounded.** Only the current step's tracks are processed, so memory does not grow with video length. Measured
cost over 1,500 videos: **0.99 ms per step** on CPU (`experiments/v4/precompute.log`).

## 5. Known limitations
- Fixed corridor regions: on curves and at intersections the "path" is approximate.
- Image-plane relations: an object far away and one close can produce the same image motion.
- Heading is not observable, so same / opposite direction are weak or disabled.
- Track id switches create a new node without history. No relation is copied across ids (tested).
