"""SafeDrive AI v4 (research track): knowledge-guided spatiotemporal risk prediction with explainable graph reasoning.

Independent of the frozen v3 system:
  * lives outside src/, pages/, ui/ (so the v3 software hash and campaign provenance are unchanged);
  * reads v3 code, caches and the v3 model bundle READ-ONLY (never writes models/ or configs/frozen_v3_hashes.json);
  * writes only to data/processed/v4_graph/, models/safedrive_v4_research/ and experiments/v4/.
"""
__version__ = "4.0.0-research"
SCENE_GRAPH_SCHEMA = "sg-1.0"
KG_SCHEMA = "kg-1.0"
FEATURE_SCHEMA = "v4feat-1.0"
