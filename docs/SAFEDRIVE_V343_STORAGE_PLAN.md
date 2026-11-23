# SafeDrive AI v3.4.3: Storage Plan for the 10-Hour Campaign

This plan is built from **measurements of this system**. Where nothing could be measured, the number is marked as a
*planning value*. Free space and encryption are **read from the machine**; they are never assumed.

## 1. Measured rates
| Item | Rate | Source |
|---|---|---|
| Full-session video (10 fps, H.264, 5-min segments) | **≈ 630–665 MB/h** | v3.3 30-min run: 330.6 MB in 0.499 h of timeline (17,979 frames); the earlier report gave 316 MiB per 30 min (≈ 630 MB/h); a 1-min session gives 665 MB/h |
| Event clip (5 s before + 5 s after) | **≈ 4.1 MB per clip** | 223 MB for 54 clips in the same run |
| Timeline (`timeline.csv`, 10 Hz) | 2.7–2.9 MB/h | same sessions |
| System log (every 5 s) | ≈ 0.2 MB/h | 720 rows/h |
| Labels + evidence frames | ≈ 5 MB/h | *planning value*: about 50 JPEG frames per hour at about 80 kB |
| Blurred shared copy | ≈ same as the raw video | *planning value*: same encoder settings; not yet measured on campaign footage |

The number of event clips depends on how many alarms there are on real roads, and that is **unknown**. Two scenarios are used:
- **Low:** 9 clips/h, which is the held-out false-alarm rate of the balanced profile.
- **High:** 108 clips/h, which is the density of the demo replay on the sensitive profile and an upper bound.

## 2. Estimate for 10 h (`readiness.storage_need`; 1 GiB = 1024 MB)
| Scenario | Video | Shared copies | Clips | Timeline, logs, labels | **Total** | **+ 30 % reserve** |
|---|---|---|---|---|---|---|
| Low clips, no sharing | 6,650 MB | 0 | 369 MB | 82 MB | 7,101 MB | **9.0 GiB** |
| High clips, no sharing | 6,650 MB | 0 | 4,428 MB | 82 MB | 11,160 MB | **14.2 GiB** |
| Low clips, with shared copies | 6,650 MB | 6,650 MB | 369 MB | 82 MB | 13,751 MB | **17.5 GiB** |
| **High clips, with shared copies (used by the audit)** | 6,650 MB | 6,650 MB | 4,428 MB | 82 MB | 17,810 MB | **22.6 GiB** |

- **Laptop data drive.** The audit requires free space ≥ the worst case (22.6 GiB ≈ 24.3 GB). Every pre-session checklist
  also requires ≥ 3 GB free.
- **Backup drive.** It holds the same data (sessions + `data/campaign/`), so it needs ≥ 22.6 GiB **free**.
  - A 64 GB drive is the practical minimum.
  - **128 GB or more** is recommended, so that the campaign can continue past 10 h and old copies can be kept.
- **Per session (60 min, worst case):** about 1.1 GB without shared copies, plus the blurred copy if one is made.

## 3. What was read from this machine (2026-11-22)
| Drive | Physical disk | Bus | Free | Encryption |
|---|---|---|---|---|
| C: | disk 0 | NVMe (internal) | ≈ 17 GB | not readable without administrator rights (Windows 11 **Home**) |
| D: (project and data) | **disk 0, the same physical disk as C:** | NVMe (internal) | ≈ 106 GB | not readable without administrator rights |
| External | — | — | **none connected** | — |

**Consequences:**
- D: has enough space for the campaign.
- C: is **not** a valid backup target: it is the same physical disk, so one failure or theft loses both copies. The backup
  tool refuses C: and D:.
- No external drive is connected, so the backup items are **BLOCKED**.

## 4. Rules during the campaign
- Run `python scripts/campaign_backup.py backup --dest <drive>:\` and then `verify` **after every driving day**. The checklist
  refuses the next session if an earlier real-driving session is missing from a verified backup.
- Do not delete raw video to save space before **all** of the following hold:
  - both annotators fully reviewed the session;
  - the backup is verified;
  - the coordinator decided to delete it. Every deletion goes through the deletion form, which logs it.
- If free space on D: drops below 30 GB, stop and plan. Nothing is deleted automatically.
- Each driving day, re-check the actual MB/h on the dashboard. If clips exceed the high scenario, revise this plan.
