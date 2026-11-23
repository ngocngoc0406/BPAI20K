"""Dynamic Traffic Scene Graph schema (sg-1.0).

All geometry is IMAGE-PLANE (normalised frame coordinates, x right, y down, origin top-left). Velocities are scene-relative image
motion after camera-motion compensation (homography), in frame units per second; scale growth is in 1/s; `tau_s` is the
image-based scale-expansion time-to-contact proxy tau = s / (ds/dt) (Lee, 1976), valid only under constant closing speed.
No physical distance, speed or time-to-collision is claimed. Unknown quantities are None, never guessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

NODE_TYPES = ("EgoVehicle", "Car", "Motorcycle", "Bus", "Truck", "Bicycle", "Pedestrian", "TrafficLight", "RoadRegion", "RiskEvent")
# COCO ids kept by the v3 detector; TrafficLight is NOT detected by it (schema slot only, never produced)
COCO_TO_TYPE = {0: "Pedestrian", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VEHICLE_TYPES = {"Car", "Bus", "Truck"}
TWO_WHEELER_TYPES = {"Motorcycle", "Bicycle"}
VULNERABLE_TYPES = {"Pedestrian", "Bicycle", "Motorcycle"}

EDGE_TYPES = ("FOLLOWING", "CUTTING_IN_FRONT_OF", "APPROACHING", "MOVING_AWAY", "CROSSING_PATH", "NEAR", "OCCLUDING",
              "SAME_DIRECTION", "OPPOSITE_DIRECTION", "POTENTIAL_CONFLICT", "INTERACTING_WITH", "LOCATED_IN")
EGO_ID = "ego"
REGION_IDS = ("region:ego_corridor", "region:left_adjacent", "region:right_adjacent")


@dataclass
class Node:
    node_id: str
    node_type: str
    timestamp: float
    object_id: int | None = None
    class_label: str | None = None
    confidence: float | None = None
    bounding_box: tuple[float, float, float, float] | None = None   # x1, y1, x2, y2 (normalised)
    center_point: tuple[float, float] | None = None
    velocity_estimate: tuple[float, float] | None = None             # scene-relative image velocity (1/s); None if history too short
    scale_growth: float | None = None                                # 1/s; None if history too short
    acceleration_estimate: tuple[float, float] | None = None         # None if history too short
    tau_s: float | None = None                                       # image-based TTC proxy; None unless approaching
    track_history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, cx, cy), bounded
    track_age_steps: int = 0
    visibility_status: str = "visible"                               # visible | partially_occluded | truncated | new
    uncertainty: float = 1.0                                         # 0 (reliable) .. 1 (unreliable)
    region: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    edge_id: str
    source_node: str
    target_node: str
    relation_type: str
    confidence: float
    timestamp: float
    evidence_features: dict[str, float]
    temporal_validity: dict[str, float]        # first_seen, last_seen, valid_since, support (m of n)
    uncertainty: float | None = None           # subjective-logic uncertainty u of the relation (lack of evidence)
    opinion: dict | None = None                # belief / disbelief / uncertainty / base rate / expected probability

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Snapshot:
    step: int
    timestamp: float
    nodes: dict[str, Node]
    edges: list[Edge]
    scene: dict[str, float]                     # frame-level context (ego motion proxies, counts)
    schema: str = "sg-1.0"

    def edges_of(self, relation: str | None = None, source: str | None = None, target: str | None = None) -> list[Edge]:
        return [e for e in self.edges if (relation is None or e.relation_type == relation)
                and (source is None or e.source_node == source) and (target is None or e.target_node == target)]

    def to_dict(self) -> dict:
        return {"step": self.step, "timestamp": self.timestamp, "schema": self.schema, "scene": self.scene,
                "nodes": {k: v.to_dict() for k, v in self.nodes.items()}, "edges": [e.to_dict() for e in self.edges]}
