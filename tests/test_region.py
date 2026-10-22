import pytest

atoms = [{"atom": "O", "bonds": ["H", "H"]}, {"atom": "O", "bonds": ["Si", "H"]}]


def test_region_no_yaml(sampler, tmp_path):
    path = tmp_path.as_posix()
    sampler.add_angle_sampling(
        name_out=path + "/test_angle",
        atoms=atoms,
        region="Box",
    )
    sampler._init_samplers(sampler.sampler_inputs, -1)
    with pytest.raises(ValueError, match="System properties must be provided for regions other than 'Box'."):
        sampler.add_angle_sampling(
            name_out=path + "/test_angle",
            atoms=atoms,
            region="Pore",
        )
        sampler._init_samplers(sampler.sampler_inputs, -1)

def test_region_with_yaml(sampler_with_system_yaml, tmp_path):
    path = tmp_path.as_posix()
    sampler_with_system_yaml.add_angle_sampling(
        name_out=path + "/test_angle_pore",
        atoms=atoms,
        region="Pore",
    )
    sampler_with_system_yaml.add_angle_sampling(
        name_out=path + "/test_angle_reservoir",
        atoms=atoms,
        region="Reservoir",
    )
    sampler_with_system_yaml.add_angle_sampling(
        name_out=path + "/test_angle_wall",
        atoms=atoms,
        region="Wall",
    )
    sampler_with_system_yaml.add_angle_sampling(
        name_out=path + "/test_angle_wall_6",
        atoms=atoms,
        region="Wall_6",
    )
    sampler_with_system_yaml.add_angle_sampling(
        name_out=path + "/test_angle_box",
        atoms=atoms,
        region="Box",
    )
    sampler_with_system_yaml._init_samplers(sampler_with_system_yaml.sampler_inputs, -1)

    with pytest.raises(ValueError, match="Invalid layer thickness specified"):
        sampler_with_system_yaml.add_angle_sampling(
            name_out=path + "/test_angle_invalid_wall",
            atoms=atoms,
            region="Wall_invalid",
        )
        sampler_with_system_yaml._init_samplers(sampler_with_system_yaml.sampler_inputs, -1)

def test_region_with_yaml_invalid_region(sampler_with_system_yaml, tmp_path):
    path = tmp_path.as_posix()
    with pytest.raises(ValueError, match="Unknown region identifier"):
        sampler_with_system_yaml.add_angle_sampling(
            name_out=path + "/test_angle_invalid_region",
            atoms=atoms,
            region="InvalidRegion",
        )
        sampler_with_system_yaml._init_samplers(sampler_with_system_yaml.sampler_inputs, -1)