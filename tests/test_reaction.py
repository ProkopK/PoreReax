atoms = [{"atom": "O", "bonds": ["H", "H"]}, {"atom": "O", "bonds": ["Si", "H"]}]


def test_reaction(sampler, tmp_path):
    path = tmp_path.as_posix()
    sampler.add_reaction_sampling(
        name_out=path + "/test_reaction",
        reactions=[(atoms[0], atoms[1])],
        dimension="Time",
        position="reactant",
    )
    sampler._init_samplers(sampler.sampler_inputs, -1)
