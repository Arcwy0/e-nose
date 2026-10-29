# Browser guide for e-nose experiments (3.3.0)

A maintainer must start the server and give you its private base URL, for
example `http://LAB_SERVER:8016`. Open `<SERVER_URL>/ui`. Florence is optional
for smell-only work; the browser cannot open USB ports or enable the heater.

## Classify one sample

1. Open **Classify**. Paste `R1`–`R17`, or all 22 values in order
   `R1`–`R17`, `T`, `H`, `CO2`, `H2S`, `CH2O`.
2. Declare the actual resistance unit matching the model. PCB exports default
   to ohms; historical legacy units must be verified, not guessed.
3. Select **Classify smell** and record prediction, confidence and OOD status.

A 17-feature model ignores environmental fields. Missing environmentals are
unmeasured defaults, not measured room conditions. PCB heater temperature is
not ambient `T`. OOD warns of a distribution mismatch; it does not identify
its physical cause or guarantee a prediction is correct.

## Preview and train a CSV

1. Supply R1–R17 and a verified label column such as `smell_label`/`Gas name`.
   Supplier Rs-mode exports are also accepted: `rs_ch0_ohm`–`rs_ch16_ohm` map
   by index to R1–R17. Physical equivalence to the old array remains unverified.
2. In **Train**, upload/paste CSV, choose units, label column and recording
   date/session/recording identity. Environmental columns are optional.
3. If labels are absent, preview still works. Add verified row annotations or
   a label column before training. Annotation rows are zero-based data rows,
   header excluded, with an inclusive start and exclusive end.
4. For continuous recordings, start with **Stable plateaus**, **Balanced RF +
   baseline**, **Log ratio to clean air**, augmentation/SNV off. Preview the
   traces, accepted/rejected windows, class counts and rejection reasons.
5. Leave **Merge historical data** on for Idea 1. Turn it off for a deliberate
   one-dataset experiment or to replace incompatible legacy history.
6. Leave **Stage candidate** on. Training then leaves the active model/history
   unchanged; the new classes will not appear as active until promotion.

Default extraction uses 60 s windows/stride, the final 25% of each analyte
exposure and a 30 s end margin. Short exposures may yield no windows. Do not
relax stability thresholds merely to force data through. Filenames, waveform
shapes and container appearance are not chemical labels.

## Evaluate, promote and roll back

In **Model Info**, select the candidate and compare it with the active model
on the **same separately collected labelled CSV**. Explicitly confirm that it
is independent of both training histories. Exact overlap is rejected, but you
must still verify independent exposures/days.

Promote only after reviewing the comparison, and retain the rollback ID.
A first deployment or missing comparison needs an explicit unevaluated override;
this is not evidence of accuracy. A known regression needs a separate override.
Rollback restores the saved model/history pair. Only use trusted artifacts.

If complete independent groups cannot be split, the UI reports **Generalization
not estimable**, with null accuracy. An empty confusion matrix is then expected;
it is not proof the model learned one class. Class distribution describes the
training data, while confusion metrics describe only a valid holdout.

## Live measurements and baseline

The sensor PC must run the [client](CLIENT_QUICKSTART.md) in live mode.
In **Live**, start polling and inspect units, freshness, device/configuration,
read/fault counters, heater state and drift. Diagnostics are read-only and may
be stale if the client stops; they are not safety interlocks.

For a baseline-relative model, allow clean air to stabilize, then select
**Set clean-air baseline**. Recapture after reconnect/session/board/configuration
changes, model reload or server restart. Client `--live-baseline` changes only
its plot. Use **Fast prediction** for provisional decisions, **Classify stable
window** for stricter measurement, and **Check recovery** before the next exposure.

## Hardware controls belong on the client

New PCB is default. The original board requires `--sensor-backend legacy`;
RLOW/menu 10 apply only to its ADC conversion. Keep RLOW constant within an
experiment. The old board has no INFO/STATUS or binary heater controls.

PCB heating defaults to supplier-configured `external` mode, which never
enables heat or changes its setpoint. Close the supplier tool before our client
uses the port. Menu 11 enables monitored PI after confirmation; menu 12 requests
disable/zero-duty verification. Target defaults to 300°C, host trip to 350°C;
these are configurable operating controls, not certified safe limits.

Both modes request controlled shutdown on client exit. Disable ramps down;
zero duty is not a cool sensor. USB loss, forced termination and host/power
failure cannot guarantee off without a firmware watchdog/interlock.
`OFF UNVERIFIED` must not be treated as off. Read [PCB safety](PCB_ACQUISITION.md)
before physical testing. Closing the browser does not stop heating.

## Experimental record

Keep operator/date, verified chemical and concentration, board/channel identity,
units/calibration (RLOW for legacy), session/exposure IDs, heater setpoint and
fault/status logs, clean-air/exposure/recovery timing, flow/distance and measured
ambient conditions. Retain original sequences, labels, image/VLM provenance,
model version and predictions privately. More rows of one exposure do not create
independent validation data.

[Client options](../enose/client/README.md) · [Setup](SETUP.md) ·
[Migration](MIGRATION.md) · [Training details](MODEL_IMPROVEMENTS.md) ·
[Testing](TESTING_GUIDE.md)
