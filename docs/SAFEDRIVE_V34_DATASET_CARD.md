# Dataset Card: SafeDrive Real-World Driving (campaign v34-realworld)

**Status 2026-11-21:**
- **0 h of real driving collected.**
- 0 eligible sessions, 0 annotated situations.
- The tooling, protocols and an empty, validated manifest exist.

This card is updated from `data/campaign/*.json` (`scripts/campaign_report.py`). Every number below comes from those files.

## Summary
| Field | Value |
|---|---|
| Purpose | Evaluate the frozen SafeDrive v3 early-warning prototype on supervised real driving: FA/h, recall by event type, lead time |
| Not for | Training or tuning during the campaign; any use outside the research team; identifying people or vehicles |
| Collection | Laptop + fixed dash-mounted camera, 10 Hz model rate, full-session video at 10 fps, timeline and system logs |
| Frozen system | v3 bundle `model_sha256` 77a3361d3ebba791… (see `configs/frozen_v3_hashes.json`) |
| Real-driving sessions / hours | **0 / 0.00 h** (target ≥ 10 h) |
| Trips / drivers | 0 / 0 |
| Labelled events (confirmed by 2 annotators) | collision 0, near_miss 0, risk_event 0 |
| Inter-annotator agreement | not available (no double-annotated session) |
| Partitions (group-wise: driver → trip → session) | train 0, dev 0, final_test 0; excluded 25 sessions (0.52 h) |
| Leakage / duplicate check | passed (trivially: no partitioned data) |

## Composition
- **Instances.** A session is a continuous recording. A situation is an annotated interval with a category, an event time and evidence.
- **Conditions** (declared by the operator, never inferred): lighting, weather, road types, traffic, camera mount.
  The value is *unknown* when not declared.
- **Excluded data:**
  - **23 legacy_unknown sessions** from before v3.3/v3.4 metadata. These have no metadata, no video, and only report files.
    Their recorded `source` field shows:
    - 13 from automated tests (pytest fixtures, 0.9 min in total);
    - 6 from a camera source with no record of what was filmed (24.5 min);
    - 4 replays of a recorded Nexar dataset clip (7.6 min, 13 model events).
    They are kept unchanged and get no metadata. They are excluded from every partition and from official metrics
    (`data/campaign/legacy_audit.json`).
  - **2 demo_video sessions** from v3.3 (a 1-min smoke test and the 30-min recording test, replaying a Nexar clip).
    They are not real driving and have no provenance.

## Collection and labelling
- Collection follows [SAFEDRIVE_V34_DATA_COLLECTION_PLAN.md](SAFEDRIVE_V34_DATA_COLLECTION_PLAN.md).
- Labelling follows [SAFEDRIVE_V34_ANNOTATION_GUIDE.md](SAFEDRIVE_V34_ANNOTATION_GUIDE.md): two blind annotators, consensus
  without forced agreement.
- Privacy follows [SAFEDRIVE_V34_PRIVACY_PROTOCOL.md](SAFEDRIVE_V34_PRIVACY_PROTOCOL.md): local only, blurred and
  human-reviewed shared copies, deletion on request.

## Known limitations (expected; to be confirmed with data)
- **One camera, one laptop, one region.** Results do not transfer to other setups without new data.
- **Rare events.** 10 h will likely give fewer than 10 confirmed events per type, so recall will be "insufficient". FA/h
  will be the main measurable outcome.
- **Few drivers** make group-wise partitions coarse. With a single driver, all data falls into one partition.
- **Blurring.** There is no plate or face detector. Shared copies rely on region blurring plus a mandatory human review,
  and the blur recall is unmeasured.
- **Night, rain, curves and specific manoeuvres** may have little or no data. Report each such case as "no data", not as a result.

## Maintenance
- `scripts/campaign_report.py` rebuilds the manifest, the validation, agreement and metric reports, and the legacy audit.
- The dashboard page **Data Collection Campaign** shows them.
- Deletions are logged without content in `data/campaign/deletion_log.jsonl`.
