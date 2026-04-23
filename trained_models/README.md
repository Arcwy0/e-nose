# trained_models/ — Classifier artifacts (not in git)

`BalancedRFClassifier.save_model()` writes joblib pickles here. The default
filename is `smell_classifier_sgd_latest.joblib` (see
`SMELL_MODEL_PATH` in `enose/config.py`).

The folder is gitignored — artifacts are regenerated from training data
on demand. If you want to share a specific trained model with a
collaborator, either:

- hand them the `.joblib` file out-of-band, or
- commit it with `git add -f path/to/model.joblib` (it will route through
  Git LFS automatically; see `.gitattributes`).

On first server start with no artifact present, an unfitted classifier
is created and the server still boots — train via the UI's **Train** tab
or `POST /smell/learn_from_csv`.
