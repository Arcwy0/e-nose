import math

import pytest

from enose.config import ENV_DEFAULTS, ENVIRONMENTAL_SENSORS
from enose.config import ENV_DEFAULTS, ENVIRONMENTAL_SENSORS
from enose.client.sensors import ENoseSensor, transform_sensor_values, validate_rlow


def test_transform_uses_supplied_rlow():
    at_one = transform_sensor_values([1000.0], rlow=1.0)[0]
    at_ten = transform_sensor_values([1000.0], rlow=10.0)[0]
    assert at_ten == pytest.approx(at_one * 10.0)


@pytest.mark.parametrize("value", [0, -1, math.inf, -math.inf, math.nan])
def test_rlow_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="greater than zero"):
        validate_rlow(value)


def test_sensor_rlow_can_change_at_runtime():
    sensor = ENoseSensor(offline_mode=True, rlow=2.0)
    sensor.set_rlow(3.5)
    assert sensor.rlow == 3.5


def test_sensor_can_parse_resistance_only_line_with_environment_defaults():
    sensor = ENoseSensor(offline_mode=False, rlow=1.0)
    sample = sensor._parse_and_transform_line(" ".join(["1000"] * 17))
    assert sample is not None
    assert len(sample) == 22
    for name in ENVIRONMENTAL_SENSORS:
        assert sample[name] == ENV_DEFAULTS[name]


def test_sensor_reads_without_environment_uart():
    class FakeSerial:
        is_open = True
        in_waiting = 1

        @staticmethod
        def readline():
            return (" ".join(["1000"] * 17) + "\n").encode()

    sensor = ENoseSensor(port=("/dev/fake", None), offline_mode=False)
    sensor.serial_conn_enose = FakeSerial()
    sensor.serial_conn_UART = None

    sample = sensor.read_single_measurement()

    assert sample is not None
    assert len(sample) == 22
    assert sample["T"] == ENV_DEFAULTS["T"]


def test_sensor_can_parse_resistance_only_line_with_environment_defaults():
    sensor = ENoseSensor(offline_mode=False, rlow=1.0)
    sample = sensor._parse_and_transform_line(" ".join(["1000"] * 17))
    assert sample is not None
    assert len(sample) == 22
    for name in ENVIRONMENTAL_SENSORS:
        assert sample[name] == ENV_DEFAULTS[name]
