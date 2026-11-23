# SafeDrive AI v3.4.4: Campaign Checklists (camera & safety, legal, privacy review, before every session)

| | |
|---|---|
| For | the operator, the supervisor, COORD-01 |
| Where | app page **14 Campaign Readiness** (tabs *Pre-session checklist* and *Camera / safety & legal checklists*), page 11 for the privacy review |
| Rule | **all of this is done while the vehicle is PARKED.** Nothing in this project is tested while driving on public roads. The prototype never replaces the driver's attention |

Every point is answered by a named person. Answers are stored in `data/campaign/attestations.jsonl` and
`data/campaign/checklists/`. The software never fills them in.

## 1. Camera & safety checklist (COORD-01, once per vehicle setup; readiness item 12)
Each point needs **confirmed + basis** (what was checked: a photo id kept locally, a measured distance, a note). A point answered
*not confirmed* keeps item 12 NOT VERIFIED.

| Point | What to check |
|---|---|
| `camera_fixed` | The camera is fixed firmly: it does not move when touched and does not fall when braking |
| `no_view_obstruction` | Camera, mount and cable do not obstruct the driver's view (windscreen wiper area, mirrors) |
| `airbag_clear` | Nothing is on or in front of an airbag cover or deployment zone (dashboard, A-pillar, steering wheel). Check the vehicle manual |
| `cable_safe` | Cables are routed and fixed; they do not hang near pedals, gear lever, steering wheel or door |
| `no_laptop_while_moving` | Nobody operates the laptop while moving; it is started and stopped only while parked; the screen is out of the driver's sight |
| `supervisor_present` | A supervisor (not the driver) is present at every session and knows the stop procedure |
| `parked_stable_run` | Camera and app ran stably while parked: camera check OK, no DEGRADED state, a session started and stopped |
| `stop_procedure` | A stop / incident procedure exists (§5) |

Item 12 is PASS only with this checklist complete **and** a camera check (page 14 → *Test camera*, 3 s, numbers only, no image) that
measured OK. A failed camera check is FAIL.

## 2. Before every real-driving session (operator + supervisor; enforced by the software)
- **6 automatic items** (measured when saved): access lists, ≥ 3 GB free, earlier real sessions in a verified external backup,
  a camera check within 60 min, frozen model / feature / threshold hashes, readiness **GO** within 7 days.
- **12 manual confirmations:** vehicle_parked, camera_fixed, camera_view, timeline_camera_ok, supervisor_present, consent_recording,
  participants_briefed, plus five new safety points: no_view_obstruction, airbag_clear, cable_safe, no_laptop_while_moving and
  stop_procedure.
- The supervisor must be a different person from the operator.
- The Live page **refuses** a `real_driving` session unless the latest checklist is complete and **less than 2 h old**, and the disk
  space and hashes still pass. Nothing is recorded otherwise.

## 3. Legal checklist (COORD-01; readiness item 13)
The software **cannot** decide legality. For each point COORD-01 answers one of:
- **confirmed + basis** (which document, rule or advice);
- **not_applicable + reason** (not allowed for `local_rules`, `consent`, `retention`, `access`);
- **needs_legal_advice**;
- **not_confirmed**.

Any *needs_legal_advice* or *not_confirmed* keeps item 13 **NOT VERIFIED** and is listed in the audit.

| Point | Question | Typically needs legal advice? |
|---|---|---|
| `purpose` | Is the purpose stated in writing (research on early collision warning, no other use)? | usually not, if written down |
| `consent` | Is consent of the driver, and information / consent of occupants, obtained where required? | **yes** if occupants are not project members |
| `faces_plates` | Is handling of third-party faces and plates defined (raw restricted, blurred copies reviewed)? | **yes**: images of identifiable people and vehicles are personal data in many jurisdictions |
| `retention` | How long is raw video kept, and when is it deleted? | **yes**: the period must be justified |
| `access` | Who may access raw video (access.json), and is it enforced (item 7)? | no (internal) |
| `sharing` | No sharing, or reviewed blurred copies to named recipients only? | yes if anything leaves the team |
| `deletion_on_request` | Can a driver or occupant have their data deleted, and is the procedure known? | usually not |
| `local_rules` | Which rules apply where the vehicle is driven (dash-camera use, personal-data law, rules on devices in vehicles)? | **always**: confirm with someone qualified for that place |

The "typically" column is guidance only. It is not legal advice and not a conclusion that recording is lawful.

## 4. Privacy review of a blurred copy (COORD-01; readiness item 11)
1. **Parked**: record a short consented `controlled_test` / `live_camera_test` session with the campaign camera, showing the scene
   in front (ideally including other parked vehicles and, with their consent, people).
2. Page 11 → create the shared (blurred) copy. The raw video stays unchanged in `<session>/video/`.
3. COORD-01 watches **every segment** and answers explicitly: *Is any face or licence plate still readable?* COORD-01 also confirms
   that the clip was recorded while parked.
   - **yes** → **REJECTED**; the copy can never be exported.
   - **no**, with every segment checked → **APPROVED**. The SHA-256 of each segment is stored.
4. If a segment changes after the review → **STALE**; export is refused and a new review is needed.
5. Every decision is appended to `data/campaign/privacy_review_log.jsonl`: reviewer, time, decision, finding, segment hashes,
   parked flag and notes.
6. Alternative: record the **no-sharing decision** on page 14. Every export is then refused, and item 11 is PASS without a blur
   review. The blur quality itself stays unverified.

Demo and Nexar videos never count. They are pre-anonymised and not campaign-camera footage.

## 5. Stop and incident procedure
- Recording is started and stopped **only while parked**. While moving, nobody touches the laptop.
- On any problem (camera falls, cable loose, app frozen, alarm noise distracting): the driver **ignores the prototype and drives
  normally**. They stop at the next safe place, and only then is the setup handled.
- If a camera or cable comes loose: park, secure it, and repeat the camera check before continuing.
- A participant may ask to stop recording or to have data deleted at any time: see the deletion procedure (page 11, coordinator).
- Any incident is written in the trip notes. The session is kept and marked; it is never silently deleted.
