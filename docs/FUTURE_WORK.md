# Future Work

1. **Scale the data.** Use all 1,500 Nexar training videos, and submit to the official test-public / test-private splits with `evaluate_submission.py`.
   Evaluate cross-dataset on DAD / CCD to measure domain shift.
2. **Physical grounding.** Monocular depth, ego-motion compensation (subtract the flow-estimated ego rotation) and camera-intrinsics
   estimation, to turn apparent τ into a metric time-to-collision estimate with its own uncertainty.
3. **Ego-involvement and agent-level risk.** Predict *which* track causes the risk (per-agent heads, graph attention) and
   evaluate that against an agent-level annotation.
4. **Stronger visual backbones.** Video encoders such as V-JEPA-style or VideoMAE features, distilled into a laptop-size student. Fine-tune the
   detector on dashcam data (night, rain).
5. **Uncertainty.** Evidential or conformal prediction for coverage-guaranteed alarm sets; detect distribution shift (night or rain)
   with the epistemic signal; per-condition calibration.
6. **Alarm policy.** Cost-sensitive threshold optimisation (missed event vs nuisance alarm), driver-reaction-time models,
   and hysteresis tuned against FA/h targets on a larger normal-driving corpus.
7. **Deployment.** ONNX / TensorRT export, Jetson-class latency study, streaming input from RTSP cameras.
8. **Human factors.** A user study of whether the grounded explanations improve driver trust and response time.
