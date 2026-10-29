# trained_models/ — Classifier artifacts (not in git)

`BalancedRFClassifier.save_model()` writes joblib pickles here. The default
filename is `smell_classifier_sgd_latest.joblib` (see
`SMELL_MODEL_PATH` in `enose/config.py`).

Runtime contents are gitignored. Share trusted artifacts privately with the
matching root `database_robodog.csv` history and pinned server environment.
Joblib deserialization can execute code; do not load untrusted model files.

Version 3.3.0 stores staged model/history/manifest pairs under `candidates/` and
deployment snapshots under `candidates/rollbacks/`. Preserve the complete tree
when migrating. Snapshot manifests bind absolute active-artifact paths, so old
rollback operations do not automatically work after moving those paths.

On first server start with no artifact present, an unfitted classifier
is created and the server still boots — train via the UI's **Train** tab
or `POST /smell/learn_from_csv`. Browser/menu training stages a candidate by
default: it becomes active only after explicit promotion. Historical merging
is a separate default-on option; direct API updates remain immediate unless
`stage_candidate=true` is requested.

See [deployment](../docs/SERVER_CLIENT_GUIDE.md#training-and-candidate-deployment)
and [migration](../docs/MIGRATION.md). Two-file replacement is recoverable, not
crash-atomic; keep independent backups. Live clean-air baselines reset on load.
