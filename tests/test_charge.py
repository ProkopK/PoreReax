import pytest

atoms = [{"atom": "O", "bonds": ["H", "H"]}, {"atom": "O", "bonds": ["Si", "H"]}]


@pytest.mark.parametrize(
    "range, expected_exception, expected_message",
    [
        (
            (1, 0),
            ValueError,
            "ChargeSampler requires a 'range' parameter as a list or tuple of "
            "two numbers",
        ),
        (
            (0, 1, 2),
            ValueError,
            "ChargeSampler requires a 'range' parameter as a list or tuple of "
            "two numbers",
        ),
        (
            "not_a_tuple",
            ValueError,
            "ChargeSampler requires a 'range' parameter as a list or tuple of "
            "two numbers",
        ),
    ],
)
def test_charge_inputs(sampler, tmp_path, range, expected_exception, expected_message):
    path = tmp_path.as_posix()
    with pytest.raises(expected_exception, match=expected_message):
        sampler.add_charge_sampling(
            name_out=path + "/test_charge",
            atoms=atoms,
            range=range,
        )
        sampler.init_samplers(sampler.sampler_inputs, -1)
