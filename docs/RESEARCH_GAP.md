# Research Gap and Contribution

## What is NOT new (stated explicitly)
- Accident anticipation from dashcam video: Chan et al. 2016, Suzuki et al. 2018, Karim et al. 2022, and others.
- Uncertainty in accident anticipation: UString (Bao et al., ACM MM 2020).
- Visual explanations for accident anticipation: DRIVE (Bao et al., ICCV 2021).
- MC-Dropout, deep ensembles, temperature scaling, Grad-CAM and Integrated Gradients as techniques.

## Gap we examine
No single, reproducible study evaluates **all of the following together** on the recent, real-world Nexar
benchmark under its horizon protocol (0.5 / 1.0 / 1.5 s before the event):

1. **Ablated contribution of interpretable image-plane motion cues** — looming / apparent τ, lateral motion
   toward the ego path, pairwise closing, optical-flow expansion — next to CNN appearance features and
   temporal context (RQ1).
2. **Earliness under a streaming alarm policy** on full videos: time-to-alert *and* false alarms per hour,
   not just window-level AP (RQ2).
3. **Whether predictive uncertainty is useful operationally**: calibration before and after temperature
   scaling, error-detection AUROC, risk–coverage, and an explicit UNCERTAIN (abstain) state in the
   risk policy (RQ3).
4. **Robustness of the entire perception → prediction chain** under controlled frame dropping, blur,
   low light, occlusion and noise, re-running detection and tracking on the corrupted pixels — and
   whether uncertainty rises when inputs degrade (RQ4).
5. **Explanations grounded in measured quantities**: Grad-CAM through the full temporal model, temporal
   Integrated Gradients, global permutation importance, and template explanations built only from
   computed motion cues (no LLM).
6. **Feasibility on commodity hardware**: a 4 GB laptop GPU, with real-time perception throughput reported.

## Contribution claim (to be supported only by the experiment files)
An integrated, reproducible pipeline — perception, motion features, temporal prediction, calibrated
ensemble uncertainty with abstention, and grounded explanation — with a complete set of baselines,
ablations (A–E), streaming early-warning metrics and robustness tests on Nexar. The contribution is
the **empirical study and the system integration**, not any single new component. Every claim in
`RESULTS_AND_DISCUSSION.md` is backed by `experiments/latest/results.json` or `robustness.json`. Where the data
does not support a hypothesis, we report the negative result.
