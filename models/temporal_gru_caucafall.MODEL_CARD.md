# CAUCAFall temporal GRU

This lightweight checkpoint is an initial shadow-mode fall classifier for the
D435i pipeline. It must not be used as the sole basis for a safety or medical
decision.

- Architecture: one-layer GRU, hidden size 32
- Inputs: speed, vertical velocity, acceleration, lean angle, posture
- Window: 10 samples at 20 Hz
- Calibrated decision threshold: `0.9775074124336243`
- SHA-256: `9596fd84e935402592e6d0e5b7f084e2c476609f38b89796aa2c0fa39e9c836d`

## Training and evaluation

The model was trained from [CAUCAFall v5](https://data.mendeley.com/datasets/7w7fccy7ky/5)
(DOI `10.17632/7w7fccy7ky.5`, CC BY 4.0). Subjects 1-7 were used for fitting,
subject 8 for early stopping and threshold calibration, and subjects 9-10 for
the untouched test set.

At the calibrated threshold, the independent test set contained 2,311 true
negative windows, 0 false-positive windows, 589 true-positive windows, and 355
false-negative windows (62.4% window recall, 0.969 ROC AUC). Six of ten staged
fall clips were detected. The negative footage is short and the camera/domain
differs from a live D435i, so these values are not production guarantees.

The checked-in D435i configuration keeps `risk.ml_weight: 0.0` and disables ML
level overrides. The checkpoint therefore logs probabilities in shadow mode
while depth-confirmed rules remain authoritative. Review D435i footage and
recalibrate before enabling ML influence.
