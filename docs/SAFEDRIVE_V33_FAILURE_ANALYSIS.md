# SafeDrive AI v3.3: Failure Analysis

This analysis looks at the deployed v3 model with the balanced profile unless stated otherwise. The model is not changed.

- **Sources:**
  - `experiments/latest/v33_benchmark_per_video.csv` and `v33_benchmark_strata.json`, produced by `scripts/realworld_benchmark_strata.py`;
  - the controlled-test results under `experiments/latest/robustness_*.json`;
  - the live-system results under `experiments/live/*.json`.
- **No real-driving failures can be analysed yet:** 0 h of real driving has been recorded.
- **Image gallery:** *Real-world Evaluation → Failure-case gallery*.

## 1. Benchmark: held-out Nexar (224 videos: 112 collision, 112 normal)
The script reproduces the published deployed result.

| Profile | Recall [95 % CI] | Hits | False alarms | FA/h [95 % CI] | Warned ≥ 1 / 2 / 3 s |
|---|---|---|---|---|---|
| balanced | 11.6 % [6.9, 18.9] | 13 / 112 | 10 | 8.7 [4.2, 16.0] | 6.2 % / 2.7 % / 2.7 % |

FA/h uses 1.151 normal-video hours, counted from the 1 s warm-up to the last step. The v3.1 report used 1.180 h,
counted from the clip duration. The alarm counts are identical.

### 1.1 Missed collisions (99 of 112)
The main failure mode is **discrimination, not timing or geometry**.

| What the deployed model did in the 5 s before the event | Missed videos |
|---|---|
| P reached the alarm threshold (0.925) but not for 3 consecutive steps (persistence rule) | 17 |
| P peaked in [0.80, 0.925): below the threshold | 50 |
| P peaked in [0.50, 0.80) | 26 |
| P stayed below 0.5 | 6 |
| *(of these)* also raised an alarm > 5 s before the event, which does not count as valid | 3 |

- The median pre-event peak in missed collisions was **0.85**. The median whole-clip peak in normal videos was **0.81**.
  On these clips the two classes overlap heavily, so a stricter or looser threshold trades recall against false alarms
  almost one-for-one: see the profile curve, 8.9 % → 17.9 % recall for 2.6 → 16.5 FA/h.
- **The warned collisions were warned late:** median lead 1.0 s. Only 3 of 13 were warned ≥ 3 s ahead
  (leads 3.7, 3.8 and 4.4 s).
- **The 17 "persistence" misses are proxy evidence for branch B (adaptive warning).** Any rule change must be selected on
  dev streams, and its false-alarm cost must be measured. They do not show that a different rule would help on held-out data.

### 1.2 False alarms (10 alarms in 8 of 112 normal videos)
| Group | Normal videos | False alarms | FA/h [95 % CI] |
|---|---|---|---|
| Urban | 49 | 8 | 15.6 |
| Sub-urban | 33 | 1 | 2.9 |
| Highway | 29 | 1 | 3.6 |
| Rain | 3 | 2 | 78: both alarms in one video, out of only 0.03 h, so no conclusion is possible |

- 7 of the 8 false-alarm videos are urban or sub-urban, i.e. dense scenes with close road users.
- 2 false alarms fired 3–5 s after the start of the stream: the model warns as soon as it is allowed to (1 s warm-up).

### 1.3 Strata by verified dataset metadata (balanced, recall [95 % CI])
| Stratum | Collision videos | Recall | Normal h | FA/h |
|---|---|---|---|---|
| Light: Normal | 101 | 12.9 % [7.7, 20.8] | 1.06 | 9.5 |
| Light: Twilight | 7 | 0 % [0, 35] | 0.07 | 0 |
| Light: Dark | 4 | 0 % [0, 49] | 0.02 | 0 |
| Weather: Clear | 71 | 15.5 % [8.9, 25.7] | 0.77 | 6.5 |
| Weather: Cloudy | 30 | 3.3 % [0.6, 16.7] | 0.36 | 8.4 |
| Weather: Rain | 11 | 9.1 % [1.6, 37.7] | 0.03 | 78 |
| Scene: Urban | 66 | 15.2 % [8.4, 25.7] | 0.51 | 15.6 |
| Scene: Highway | 32 | 3.1 % [0.6, 15.7] | 0.28 | 3.6 |
| Scene: Sub-urban | 9 | 11.1 % [2.0, 43.5] | 0.35 | 2.9 |

- **Dark / twilight:** there are only 11 collision videos, so performance there is **unknown**, not "worse".
- **Highway and cloudy recall** is low (1 of 32, 1 of 30), but the intervals overlap the overall value. This is a hypothesis to
  check with real data, not a finding.

## 2. H3: the fixed image-plane corridor
**H3:**
- The "driving corridor" used by the corridor cues and audio messages is fixed in the image (|cx − 0.5| < 0.2, cy > 0.35).
- It therefore ignores curves, turns and the lane geometry.
- The hypothesis is that this causes missed or wrong warnings in curves and turns.

There are no map, lane or steering data, so H3 is measured with **two proxies**.

| Proxy | Groups | Collision videos | Recall (balanced) [95 % CI] | Normal videos | FA/h |
|---|---|---|---|---|---|
| Turn proxy: median \|horizontal camera shift\| (tertiles fixed on dev: 0.0076 / 0.0157 frame widths/s) | low | 34 | 8.8 % [3.0, 23.0] | 38 | 2.5 |
| | mid | 34 | 14.7 % [6.4, 30.1] | 40 | 16.4 |
| | high (turning / curving) | 44 | 11.4 % [5.0, 24.0] | 34 | 6.1 |
| Largest road user in the last 2 s before the collision | inside the fixed corridor | 47 | 12.8 % [6.0, 25.2] | — | — |
| | outside the fixed corridor | 65 | 10.8 % [5.3, 20.6] | — | — |

**Result:** recall when the camera is turning, and recall when the largest road user is outside the fixed corridor, are not
measurably lower. All intervals overlap, and the median turn proxy is the same for warned and missed collisions (0.0124 vs 0.0126).

What this does and does not tell us:
- On this benchmark, **H3 is not the dominant failure cause**. Discrimination is (§1.1).
- It does not rule out H3. With 34 to 65 videos per group, a difference of about 10 points cannot be detected.
- The proxy is not road curvature.
- Corridor-dependent outputs (the "vehicle ahead" message, the `lead_in_corridor` log) can still be wrong in curves. This
  needs real-driving sessions with the `curves` road type and `curve` tags.

## 3. Score overlap and temporal consistency (held-out 224, clean)
Source: `experiments/latest/v33_temporal_overlap.json`, produced by `scripts/v33_temporal_overlap.py`.

| Measure | Collision videos | Normal videos |
|---|---|---|
| Peak P (10 / 25 / 50 / 75 / 90 %): collision = 5 s before the event, normal = whole clip | 0.60 / 0.78 / 0.86 / 0.93 / 0.96 | 0.51 / 0.68 / 0.81 / 0.88 / 0.93 |
| Mean absolute change of P between 0.1 s steps | 0.063 | 0.053 |
| Runs above the balanced threshold that last < 3 steps (removed by the persistence rule) | 66 % of 77 | 65 % of 31 |

- **AUC = 0.63** when the 5 s pre-event peak is used to separate collision videos from normal videos.
  - The AUC of the clip-mean P is 0.73, but that is not usable causally.
  - Collision videos also peak at 0.79 (median) *more than 5 s before* the event, which is too early to count as a valid warning.
- **The output is jittery.**
  - P changes by about 0.05 to 0.06 every 0.1 s.
  - Normal videos cross the threshold 0.89 times per minute, about 54 per hour before the persistence rule. After the rule, 8.7 FA/h remain.
- **The persistence rule removes short runs at the same rate in both classes** (65 % vs 66 %).
  - Relaxing or adaptively changing the rule would add false alarms roughly in proportion to the recall it gains.
  - The 17 "persistence misses" of §1.1 are therefore **not** evidence that a smarter warning rule alone would separate the classes.

## 4. Lighting-related errors
### 4.1 Verified dataset lighting (benchmark, §1.3)
- There are only 4 dark and 7 twilight collision videos, and none was warned at the balanced profile. The intervals are 0–49 % and 0–35 %.
- **Lighting-related performance on real night driving is unknown.**

### 4.2 Controlled rapid light change (tunnel-like exposure swings)
`light_change` switches the image gain between 1.0 and dark / bright levels every 2.5–4.5 s, with a 0.3 s ramp.
- Severity 1: dark gain 0.35.
- Severity 2: dark gain 0.15, plus over-exposure with gain 1.8.

**(a) Window level** (`robustness_v3_light_change.json`; the same 60 test videos and 180 windows as the original robustness study)

| Condition | mAP | ROC-AUC | F1 | ECE | Mean MI |
|---|---|---|---|---|---|
| clean | 0.805 | 0.905 | 0.828 | 0.093 | 0.0164 |
| light_change 1 | 0.695 | 0.870 | 0.734 | 0.114 | 0.0204 |
| light_change 2 | 0.657 | 0.837 | 0.680 | 0.135 | 0.0225 |

- The clean result reproduces v3.1 exactly.
- **Caveat:** these clips are only about 4 s long, and the schedule makes its first change after 2.5–4.5 s. Only the end of each
  clip is exposed, so this test under-states exposure.

**(b) Full streams with the deployed alarm logic, paired with a clean re-extraction of the same 60 videos**
(`v33_light_change_stream.json`; 30 collision / 30 normal videos, 0.30 normal h)

The re-extracted clean streams are identical to the original caches (Δ = 0), so the baseline is exact.

| Condition | Detections / step | Steps without detection | Mean P shift per video [95 % CI] | Mean P before event (collision) | Mean P (normal) | Recall balanced [95 % CI] | FA balanced |
|---|---|---|---|---|---|---|---|
| clean | 3.25 | 14.2 % | — | 0.470 | 0.263 | 6.7 % [1.8, 21.3] (2/30) | 1 |
| light_change 1 | 3.10 | 15.4 % | −0.029 [−0.037, −0.022] | 0.422 | 0.249 | 3.3 % [0.6, 16.7] (1/30) | 0 |
| light_change 2 | 2.91 | 17.8 % | −0.039 [−0.049, −0.028] | 0.405 | 0.251 | 3.3 % [0.6, 16.7] (1/30) | 1 |

Paired comparison against clean, balanced profile:
- Severity 1: 0 warned collisions gained, 1 lost (McNemar exact p = 1.0).
- Severity 2: 0 gained, 1 lost (p = 1.0).
- Sensitive profile: 2 lost at each severity (p = 0.5).

What this shows:
- **Measured effects (95 % CIs exclude zero):**
  - detections drop by 4.5 % and 10.3 % per step;
  - the share of steps without any detection rises;
  - P falls, and it falls more for collision videos (−0.048 / −0.065) than for normal ones (−0.014 / −0.012).
    The class separation therefore shrinks.
- **Not shown:** whether alarm recall or FA/h changes.
  - Even the clean streams warn only 2–3 of the 30 collisions, so no change can be detected.
  - **The model is not shown to be robust to rapid light change.** The evidence points to degradation, through fewer detections
    and a lower, less separable P.

## 5. Controlled system tests (live pipeline)
| Test | Result | Source |
|---|---|---|
| 30 min continuous, demo replay, event clips **and** full-session video (v3.3) | **Passed.** 1800.6 s, alive, no error, 9.98 steps/s, 54/54 clips with 100/100 frames, full video 17,979 frames (= timeline steps, 0 dropped), 0 missing steps | `experiments/live/demo_30min_record.json` |
| Camera disconnect / reconnect | Passed (v3.2) | `cam_disconnect.json` |
| Camera used by another app | DEGRADED INPUT shown, outputs suppressed (v3.2) | `cam_occupied.json` |
| Near-full disk (simulated) | Event logged without a clip, reason recorded, no fake clip (v3.2) | `demo_disk_full.json` |
| Camera shake / occlusion | Window-level degradation measured (v3.1); v3-stabilized helps shake only partially | `robustness_v3*.json` |

Failure-relevant details of the 30-minute run:
- **Timing.** 23 % of steps were later than their 0.1 s slot (v3.2 event-only 5-min run: 19 %).
  - The p95 step time was 92 ms, close to the 100 ms budget.
  - The full-video encoder and the clip writer share the CPU with inference. On a slower machine this would show up as DEGRADED INPUT.
- **Memory.**
  - RSS was 3039 MB at the start, peaked at 3912 MB around 7.5 min, then fell back: 2853 MB at 15 min, 2984 MB at the end.
  - The slopes per quarter were +11, −44, +1 and +25 MB/min.
  - This is non-monotonic, so there is **no evidence of a leak over 30 min**. A leak smaller than these swings cannot be excluded
    without a longer run.
- **GPU memory** rose from 1085 MB to a maximum of 1349 MB. The maximum is the same as in the v3.2 5-min run.

## 6. Real driving
**0 h recorded.** Nothing can be said yet about:
- the real-world FA/h, repeated warnings or warnings without a hazard;
- recall or lead time for real cut-ins, lead braking or lane changes;
- curves, many vehicles, or the camera mount.

The tooling to measure all of these exists:
- the timeline;
- annotation with a second annotator;
- the metrics;
- the dashboard tab ④.

## 7. Summary of failure causes (ranked by evidence)
1. **Weak discrimination** (strong evidence): a pre-event peak AUC of 0.63, heavily overlapping peaks, and 99 of 112 collisions missed.
2. **Late or early timing** (strong evidence): median lead 1.0 s; collision videos often peak more than 5 s before the event.
3. **Jittery output** (measured): it affects positives and negatives alike, so it is not by itself a separability problem.
4. **Lighting** (controlled evidence of degradation; real night behaviour unknown).
5. **Fixed corridor (H3)** (no measurable effect on the benchmark proxies; not ruled out).
