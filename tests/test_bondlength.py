import pytest


bonds = [{"bond": "Si-O", "bonds_A": ["O", "O", "O"], "bonds_B": ["H"]}]


@pytest.mark.parametrize(
    "dimension, num_bins, range, expected_exception, expected_message",
    [
        (
            "not_a_dimension",
            10,
            (0, 1),
            ValueError,
            "BondLengthSampler does not support dimension",
        ),
        (
            "Bond Length",
            -10,
            (0, 1),
            ValueError,
            "BondLengthSampler requires a positive integer 'num_bins' parameter.",
        ),
        (
            "Bond Length",
            10,
            (1, 0),
            ValueError,
            "BondLengthSampler requires a 'range' parameter as a list or tuple of two numbers",
        ),
        (
            "Bond Length",
            10,
            (0, 1, 2),
            ValueError,
            "BondLengthSampler requires a 'range' parameter as a list or tuple of two numbers",
        ),
        (
            "Bond Length",
            10,
            "not_a_tuple",
            ValueError,
            "BondLengthSampler requires a 'range' parameter as a list or tuple of two numbers",
        ),
    ],
)
def test_bond_length_inputs(
    sampler, tmp_path, dimension, num_bins, range, expected_exception, expected_message
):
    path = tmp_path.as_posix()
    with pytest.raises(expected_exception, match=expected_message):
        sampler.add_bond_length_sampling(
            name_out=path + "/test_bond_length",
            bonds=bonds,
            dimension=dimension,
            num_bins=num_bins,
            range=range,
        )
        sampler.init_samplers(sampler.sampler_inputs, -1)
