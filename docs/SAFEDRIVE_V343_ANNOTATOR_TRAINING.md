# SafeDrive AI v3.4.3: Annotator Training

**Purpose:** before the campaign, prepare the two annotators (ANN-A, ANN-B) and test the labelling workflow on a practice pack.

**Status:**
- The pack is built.
- **Nobody has been trained yet.** Item 14 is NOT VERIFIED until both annotators complete the pack and meet the criteria.

> **TRAINING DATA — NOT REAL DRIVING — NEVER EVALUATION DATA.**
> The pack lives in `data/annotator_training/sessions/`, **outside** `data/sessions/`. It can therefore never enter:
> - the campaign manifest or partitions;
> - the agreement report or `campaign_metrics.json`;
> - the real-driving hours.
>
> Training results only describe the annotators' readiness.

## 1. The pack (`TRAINING_PACK_01`, built by `training.build_pack()`, seed 7)
| | |
|---|---|
| Source | Nexar dashcam clips from the **train split** (never held-out). They are already anonymised by Nexar |
| Video | one video, 99 s at 10 fps and 640×360: 9 clips, separated by 1 s of black |
| Event clips (4) | 8 s before and 2 s after the Nexar event time. Accepted labels: collision / near_miss / risk_event. Reference event time = the Nexar `time_of_event` |
| Normal clips (3) | 10 s of negative videos. Accepted label: normal |
| Degraded clips (2) | Event clips made very dark and motion-blurred (low_light severity 2 + motion_blur severity 2). Expected answer: **uncertain**, unless the event is clearly visible |
| Hidden key | `training_key.json`: clip windows, accepted labels, reference times. It is shown **only** to the coordinator, and only after both annotators have finished |

**Known limitation.** Nexar does not separate collisions from near misses, so the key accepts all three event categories.
The coordinator discusses the choice of category, using the definitions in
[SAFEDRIVE_V34_ANNOTATION_GUIDE.md](SAFEDRIVE_V34_ANNOTATION_GUIDE.md). Category agreement is reported, but is not a pass
criterion. *risk_event* has no ground truth in this pack.

## 2. Procedure
1. **Read the guide.** Each annotator reads the annotation guide alone. The coordinator explains the five categories, the
   event time and the evidence rules.
2. **Label blind.** Annotators work separately, each under their **own** pseudonym:
   - on page 11, choose workspace **Annotator training**, session `TRAINING_PACK_01`, role *Annotator*;
   - label every situation in the video. Every event label needs an event time, an evidence text and evidence frame times
     (enforced by the software);
   - use *normal* for stretches with nothing notable, and *uncertain* when the evidence is insufficient;
   - press **"I watched the WHOLE session"** at the end.
   - Blind mode shows each annotator only their own labels: no model output, no other annotator's labels and no key.
3. **Coordinator review.** COORD-01 opens the same session as *Coordinator*. When both annotators are done, the page shows:
   - the result for each annotator against the key;
   - the agreement between the two: κ, event-vs-not κ, and start / event / end offsets;
   - the key itself;
   - the consensus table and the disagreement form.

   Record disagreements, but **never force agreement**.
4. **Debrief.** Discuss every miss, every false event and every large timing offset. If an annotator does not pass, give a
   second practice round. Build a second pack with `build_pack(seed=…, out=…)`: the pack is built once per folder.

## 3. Pass criteria (`training.CRITERIA`)
**Per annotator:**

| Criterion | Threshold |
|---|---|
| Watched the whole pack | marked fully reviewed |
| Event detection | ≥ 80 % of the 4 event clips have an event label inside the clip |
| Timing | median of \|labelled event time − reference\| ≤ 1.0 s |
| False events on normal clips | ≤ 1 |
| Categories practised | each of collision, near_miss, risk_event, normal, uncertain used at least once |
| Evidence | every event label has a description and evidence frames |

**Pair:** event-vs-not κ between ANN-A and ANN-B ≥ 0.60.

When both annotators pass and the pair criterion holds, the page writes `data/annotator_training/training_report.json` with
`passed = true`, and readiness item 14 becomes PASS.

**Why "used every category at least once":** the brief asks for practice of all five. A trained annotator labels only what is
there, so the coordinator points to suitable moments: for example a short risky moment inside a "normal" clip, or the degraded
clips for *uncertain*.

## 4. What this training cannot show
- How good the labels will be on **real campaign footage**, with a different camera, local traffic and motorcycles.
- Whether the two people really work independently. The software keeps them blind to each other, but cannot stop them from
  sharing a name or talking.

**The first real sessions** therefore get extra coordinator spot checks. Agreement on campaign data is reported separately,
in `agreement_report.json`, and never mixed with training results.
