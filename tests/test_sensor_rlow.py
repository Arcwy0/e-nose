import math

import pytest

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
