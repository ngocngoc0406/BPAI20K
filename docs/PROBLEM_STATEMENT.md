# Problem Statement

## Context
Rear-end, cut-in and vulnerable-road-user conflicts typically develop over 1–3 seconds. A system that warns
the driver even half a second earlier gives them time to react. Dashcams are cheap and widely installed,
but turning their video into **trustworthy** early warnings is hard:
- the cues are subtle: an object growing in the image, a lateral drift into the ego path, two vehicles converging;
- a false alarm costs driver trust, and a missed event can cost a life;
- a black-box score gives the driver no reason to believe it, and no way to tell when it is guessing.

## Task
Given a monocular dashcam video stream, at every time *t* use **only frames up to *t*** to estimate the
probability that a collision or near-collision involving the ego vehicle happens shortly after *t*. Also:
1. quantify the uncertainty of that estimate, and abstain (**UNCERTAIN**) when it is unreliable;
2. map it to a graded risk level (NORMAL / CAUTION / WARNING / HIGH RISK);
3. explain the warning with evidence the system actually computed.

## Formal definition
Let a video be sampled on a 10 Hz grid, with frames x₁…x_T. Let τ_e be the annotated event time (for positive videos).
For an observation window of length W ending at t, the model outputs p(t) = P(event in (t, t+H] | x_{t−W..t}).
Following the Nexar protocol, positive windows end at t = τ_e − h with h ∈ {0.5, 1.0, 1.5} s, and negatives come from
normal-driving videos. Nothing at or after τ_e is ever observed.

## Constraints
- A single laptop: RTX 3050 4 GB, 16 GB RAM, Windows 11.
- Real, licensed data (Nexar Open Data License). No synthetic labels.
- Every result reported in the app or the manuscript must come from experiment files.

## Out of scope
Physical time-to-collision (it needs calibration and depth), vehicle control, certification as a safety device.
