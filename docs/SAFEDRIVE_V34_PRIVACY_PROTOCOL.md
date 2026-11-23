# SafeDrive AI v3.4: Privacy Protocol

## 1. Principles
- **Everything stays local.** Video is processed and stored on the collection computer, and the software uploads nothing.
  Do not copy raw video to any cloud, chat or external service.
- **Collect only what the research needs.**
  - Drivers are pseudonyms (D01, …); real names are refused.
  - Location is optional free text and must not contain addresses.
  - No audio is recorded.
- **Consent first.** A session cannot start without driver consent, informed occupants (for full video), a supervisor,
  setup before moving and confirmation that local recording rules were checked. The software enforces this.

## 2. Legal and consent check (before the first session)
The operator must check, and record that they checked (`regulations_checked`):
- the local rules on in-vehicle and dash-camera recording of public roads;
- the rules on processing personal data, which covers images of identifiable people and licence plates.

This document is **not legal advice**. If the rules are unclear, do not record.

Consent covers three things:
- **Driver:** recording; storing the raw video locally for annotation; (optional `share_consent`) sharing a blurred,
  privacy-reviewed copy within the research team.
- **Occupants:** informed before a full-session recording.
- **Other road users:** they cannot consent. They are protected by data minimisation, blurring before any sharing, and
  local-only storage.

## 3. Two tiers of video
| Tier | Where | Who | Content |
|---|---|---|---|
| **Raw** | `data/sessions/<id>/video/` | names on `raw_access` in `data/campaign/access.json` (annotators who need full detail, e.g. brake lights) | unblurred, except that people are blurred at recording time when `blur_people` is on |
| **Shared** | `data/sessions/<id>/shared/` | the research team, only after an approved review | blurred copy |

Access gating:
- In the app it is name-based: the annotation page shows raw video only to people on the raw-access list, and shared video only when approved.
- **This is application-level gating on a shared computer, not security.**
  - Protect the data folder with Windows account permissions.
  - Use full-disk encryption (BitLocker) on the collection laptop.
  - Keep the access list short.

## 4. Blurring before sharing (`src/realworld/privacy.py`)
No face or licence-plate detector exists in this project. The shared copy is therefore made by **conservative, region-based
blurring** with the COCO detector, after the trip:
- **Detector.** Run on every frame at a **low confidence (0.15)**. Boxes are kept for ±2 frames to bridge detector flicker.
- **What is blurred:**
  - **persons:** the whole box (covers faces);
  - **bicycles / motorcycles:** the whole box (riders, small plates);
  - **cars / buses / trucks:** the **whole box**. Since the 2026-11-22 audit, the earlier lower-45 % band missed rear plates
  mounted at mid-height.
- **every box** is padded by 8 % on each side.
- **Side effect:** large false detections, e.g. the own hood, can blur most of a frame, so shared copies may be heavily blurred.
- **Logging.** `blur_log.json` records the method and the number of blurred boxes per segment.

Limitations, measured status: **blur recall has not been measured.**
- Missed or very small detections, plates mounted high, faces seen through windows, and roadside signs with personal data can remain visible.
- **The human review is therefore mandatory.**

## 5. Mandatory human review
- **Who:** a coordinator on the `coordinators` list.
- **How:** watch every shared segment completely, then save the review on the annotation page. Approval requires every
  segment to be ticked as checked.
- **Reject if** any face or plate is readable. Then either re-create the copy, or do not share the session.
- **Re-creating** a shared copy deletes the earlier review automatically.
- **Approval** is recorded in `shared/privacy_review.json` (reviewer, segments, date, notes).
- **Only approved copies are "shareable".** The manifest records the privacy status of every session.
- **Export.** Sharing goes only through **Export** on page 11 (`privacy.export_shared`). It works only for:
  - a coordinator;
  - an approved copy that is unchanged since the review.
  It copies the blurred segments only, never raw video, verifies SHA-256 and logs to `data/campaign/export_log.jsonl`.
  Copying files by hand bypasses this and is not allowed.

## 6. Deletion on request
- **Who can delete:** a driver or occupant can ask for deletion at any time.
- **How:** on the annotation page (coordinator) → *Delete this trip's data*, typing the session id. This removes:
  - the whole session folder: raw and shared video, timeline, annotations, evidence frames, report;
  - its event clips and metadata in Near-Miss Memory.
- **Audit record:** `data/campaign/deletion_log.jsonl` keeps only the session id, the time, who asked (pseudonym or role),
  the reason and the counts. It keeps no content.
- **Afterwards:** re-run `scripts/campaign_report.py`. The partition assignment of a deleted group stays frozen, which
  prevents a later re-split.

## 7. What must never happen
- Uploading raw or unreviewed video to any external service.
- Storing real names, phone numbers, addresses or plate numbers in metadata, notes or annotations.
- Sharing a shared copy whose review is missing or rejected.
- Using the data for anything other than the stated research purpose.
