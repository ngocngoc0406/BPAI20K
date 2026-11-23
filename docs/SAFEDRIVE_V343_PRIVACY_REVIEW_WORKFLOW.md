# SafeDrive AI v3.4.3: Privacy Review Workflow

This workflow extends [SAFEDRIVE_V34_PRIVACY_PROTOCOL.md](SAFEDRIVE_V34_PRIVACY_PROTOCOL.md). The code is in `src/realworld/privacy.py`.

**Current state:** item 11 is **NOT VERIFIED**.
- No blurred copy of campaign-camera footage exists or has been reviewed.
- The blur has **not** been validated on real footage, and **no miss rate is claimed**.

## 1. Two separate copies
| Copy | Folder | Changed by review? | Who sees it |
|---|---|---|---|
| Raw | `data/sessions/<id>/video/` | **Never**. Review, rejection and revocation never touch it | names on `raw_access` |
| Shared (blurred) | `data/sessions/<id>/shared/` | Replaced only by *re-create* | the team, **only** once it is approved and sharing is enabled |

## 2. States of a shared copy (`privacy.shared_status`)
```
(no copy) --create--> pending_review --review: nothing readable, all segments--> approved --export--> (exported, logged)
                          |                                                       |   \
                          +--review: something readable / incomplete--> rejected  |    +--any file changed--> stale (review again)
                                                                                  +--revoke--> revoked (no export possible)
re-create (any state) --> pending_review   (the old review and revocation are removed; history stays in the logs)
```
| State | Export allowed? |
|---|---|
| `pending_review` | no |
| `approved` | yes: by a coordinator, while sharing is enabled |
| `rejected` | no |
| `stale` | no: a file changed after the review, or the reviewer is no longer a coordinator |
| `revoked` | no |

## 3. Review rules (enforced by the software)
1. **Only a coordinator** on the `coordinators` list (now COORD-01) can review. Annotators and unknown names are refused.
2. The reviewer must tick **every** segment as watched completely.
3. The reviewer must answer explicitly whether **any face or licence plate is readable**:
   - *No* is the only answer that can approve.
   - *Yes* records the copy as **rejected**, whatever else was ticked.
   - *Not answered* cannot approve.
4. The review stores the reviewer, the time, the decision, the notes and the **SHA-256 of every segment**. Any later change of a
   segment makes the review **stale**.
5. There is no automatic approval and no way to skip the review: `export_shared` checks the state itself.

## 4. Export (the only sanctioned way to share)
On page 11, the coordinator uses the export form (`privacy.export_shared`). Export is refused when:
- the caller is not a coordinator;
- sharing is disabled;
- the copy is not `approved` (pending, rejected, stale or revoked).

It copies only the blurred segments plus the review and blur logs, re-hashes every file, and logs the export to
`data/campaign/export_log.jsonl`. Raw video is never exported.

## 5. Revocation (blur found insufficient after sharing)
On page 11 the coordinator uses *Revoke this shared copy* (`privacy.revoke_shared`), giving a reason. This:
- marks the copy **revoked**, which blocks any further export;
- lists every earlier export of the session from the export log;
- optionally deletes the exported folders that are still reachable and carry a matching `export_manifest.json`;
- records everything in `data/campaign/revocation_log.jsonl`.

**Limitation:** software cannot recall copies that have already been passed on to other people. The log names the export
destinations, so the coordinator can contact the holders. A new blurred copy can then be created and reviewed again.

## 6. No-sharing decision (an alternative to validating the blur now)
- On page 14 → *Coordinator confirmations*, the coordinator can set sharing to **disabled**, with a reason.
- This is the written decision that **nothing is shared** until the blur has been validated on campaign-camera footage.
- While it is in force, `export_shared` refuses **every** copy. The audit proves this on a throw-away copy each time it runs.
- Readiness item 11 is then PASS in the mode "sharing disabled". The evidence still says the blur quality is unverified.
- Only a coordinator can enable sharing again. The policy file keeps the history.

## 7. Validating the blur on campaign-camera footage (COORD-01)
Do this once before the first shared copy, and again if the camera, the mount or the lighting changes a lot.

1. **Record a parked clip.** Parked, with consent from everyone visible, record a short (1–3 min) session with the campaign
   camera:
   - data kind *live camera test*, or *controlled test*;
   - at least one parked vehicle with its rear plate visible at 3–10 m;
   - one consenting person walking past.
2. **Make the blurred copy.** On page 11 → Coordinator → *Create the blurred shared copy*.
3. **Review it.** Watch every segment completely. Pause on the closest plate and face moments, and use frame-by-frame where needed.
4. **Decide.**
   - If **anything** is readable, answer *Yes* (the copy is rejected) and note where. Then either:
     - stop: keep the no-sharing decision; or
     - ask the developer for a stronger blur. This is a software change: it needs a changelog entry, and only `privacy.py`
       changes; the model is not touched.
   - If nothing is readable, approve it. Readiness item 11 becomes PASS for campaign-camera footage.

**What does not count:**
- Demo, Nexar and test videos **never** count. Nexar clips are already anonymised, and a test video cannot stand in for the
  real camera, so readiness ignores approved copies of `demo_video` sessions.
- Even after approval, **the blur miss rate is not measured**. Every shared copy of real driving still needs its own complete
  review.

## 8. Tests (in `tests/test_v343_readiness.py`)
| Test | What it checks |
|---|---|
| `test_unreviewed_copy_is_pending_and_never_exported` | an unreviewed copy is `pending_review`; approval needs the explicit finding |
| `test_readable_face_or_plate_rejects_copy_and_revocation_blocks_export` | rejection; export, then revocation, with the export deleted and the raw video untouched; the stale state |
| `test_unauthorised_people_are_refused` | annotators and unknown names cannot review, attest, change the policy, export or revoke |
| `test_no_sharing_decision_blocks_every_export_and_demo_never_validates_blur` | the no-sharing decision blocks export; an approved demo copy does not validate the blur |
