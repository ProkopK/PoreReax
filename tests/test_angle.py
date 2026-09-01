import pytest

atoms = [{"atom": "O", "bonds": ["H", "H"]}, {"atom": "O", "bonds": ["Si", "H"]}]


@pytest.mark.parametrize(
    "num_bins, angle, expected_exception, expected_message",
    [
        (
            -10,
            "all",
            ValueError,
            "AngleSampler requires a positive integer 'num_bins' parameter.",
        ),
        (
            10,
            ["Si", "O"],
            ValueError,
            "AngleSampler requires 'angle' parameter to be a string",
        ),
        (
            10,
            "Si-O-O-O",
            ValueError,
            "AngleSampler 'angle' parameter must be 'all' or in the format 'A-B-C'",
        ),
        (
            10,
            "Si-O-Z",
            ValueError,
            "AngleSampler 'angle' parameter contains unknown atom identifier 'Z'",
        ),
    ],
)
def test_angle_inputs(
    sampler, tmp_path, num_bins, angle, expected_exception, expected_message
):
    path = tmp_path.as_posix()
    with pytest.raises(expected_exception, match=expected_message):
        sampler.add_angle_sampling(
            name_out=path + "/test_angle",
            atoms=atoms,
            num_bins=num_bins,
            angle=angle,
        )
        sampler.init_samplers(sampler.sampler_inputs, -1)
