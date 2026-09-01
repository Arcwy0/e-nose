# Browser guide for e-nose experiments

This guide is for running smell classification and training without editing
Python code. A project maintainer must first start the server and give you its
URL, for example `http://192.168.1.42:8080`.

## Open the interface

Open `<SERVER_URL>/ui` in a browser. The status area should report that the smell
classifier is loaded. Florence-2 vision is optional for smell-only work.

## Classify one 22-feature sample

1. Open **Classify**.
2. Paste 22 comma-separated values in this order: `R1`–`R17`, `T`, `H`, `CO2`,
   `H2S`, `CH2O`.
3. Select **Classify**.
4. Record the predicted class, confidence, and OOD status. Treat an OOD warning
   as an indication that the sample differs from the training distribution.

## Train from a CSV

1. Keep one row per measurement.
2. Provide the 22 sensor columns and a label column such as `Gas name` or
   `smell_label`.
3. Open **Train**, upload or paste the CSV, choose its label column, and start
   training.
4. Open **Model Info** and verify the known classes and evaluation metrics.

Do not compare accuracy from random rows of the same recording as though it were
a new-day test. For drift research, keep session/day identifiers and follow
[`DRIFT_DATA_PROTOCOL.md`](DRIFT_DATA_PROTOCOL.md).

## Watch live measurements

The computer connected to the e-nose must run the client in live mode. Once it
is publishing, open **Live** and select **Start polling**. The plots show the 17
resistance channels, environmental values, predictions, and drift diagnostics.

`RLOW` is a hardware calibration used when the client converts raw ADC values to
resistance. The operator can set it when starting the client or through menu
option 10. Keep it constant within an experiment and write it in the laboratory
record; changing it rescales all resistance channels.

## Recommended experimental record

For every recording, retain:

- date, operator, object/odour label, and session identifier;
- e-nose serial numbers and `RLOW`;
- exposure, settling, recording, and purge durations;
- temperature, humidity, and relevant room conditions;
- raw/session data, model version, prediction, confidence, and OOD status;
- the image and VLM-derived label for autonomous Idea-1 training.

## Where to go next

- [`CLIENT_QUICKSTART.md`](CLIENT_QUICKSTART.md): client commands and scenarios.
- [`TESTING_GUIDE.md`](TESTING_GUIDE.md): complete staged validation.
- [`MIGRATION.md`](MIGRATION.md): moving the installation to another PC.
- [`SERVER_CLIENT_GUIDE.md`](SERVER_CLIENT_GUIDE.md): API and deployment details.
