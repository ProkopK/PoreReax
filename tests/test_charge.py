import pytest


atoms = [{"atom": "O", "bonds": ["H", "H"]}, {"atom": "O", "bonds": ["Si", "H"]}]


@pytest.mark.parametrize(
    "num_bins, range, expected_exception, expected_message",
    [
        (
            -10,
            (0, 1),
            ValueError,
            "ChargeSampler requires a positive integer 'num_bins' parameter.",
        ),
        (
            10,
            (1, 0),
            ValueError,
            "ChargeSampler requires a 'range' parameter as a list or tuple of two numbers",
        ),
        (
            10,
            (0, 1, 2),
            ValueError,
            "ChargeSampler requires a 'range' parameter as a list or tuple of two numbers",
        ),
        (
            10,
            "not_a_tuple",
            ValueError,
            "ChargeSampler requires a 'range' parameter as a list or tuple of two numbers",
        ),
    ],
)
def test_charge_inputs(
    sampler, tmp_path, num_bins, range, expected_exception, expected_message
):
    path = tmp_path.as_posix()
    with pytest.raises(expected_exception, match=expected_message):
        sampler.add_charge_sampling(
            name_out=path + "/test_charge",
            atoms=atoms,
            num_bins=num_bins,
            range=range,
        )
        sampler.init_samplers(sampler.sampler_inputs, -1)
