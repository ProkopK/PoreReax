atoms = [{"atom": "O", "bonds": ["H", "H"]}, {"atom": "O", "bonds": ["Si", "H"]}]


def test_density(sampler_with_system_yaml, tmp_path):
    path = tmp_path.as_posix()
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_time",
        atoms=atoms,
        dimension="Time",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_cartesian_1d",
        atoms=atoms,
        dimension="Cartesian1D",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_cartesian_2d",
        atoms=atoms,
        dimension="Cartesian2D",
        direction="xy",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_pore_1d",
        atoms=atoms,
        dimension="Pore1D",
        direction="r",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_pore_1d",
        atoms=atoms,
        dimension="Pore1D",
        direction="p",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_pore_1d",
        atoms=atoms,
        dimension="Pore1D",
        direction="d",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_pore_2d",
        atoms=atoms,
        dimension="Pore2D",
        direction="rp",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_pore_2d",
        atoms=atoms,
        dimension="Pore2D",
        direction="rz",
    )
    sampler_with_system_yaml.add_density_sampling(
        name_out=path + "/test_pore_2d",
        atoms=atoms,
        dimension="Pore2D",
        direction="pz",
    )
    sampler_with_system_yaml._init_samplers(sampler_with_system_yaml.sampler_inputs, -1)
