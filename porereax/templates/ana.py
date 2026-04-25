import porereax as prx

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')


# TODO replace a, b, c, ... with your atom names from atom_lib

num_sims = {{ NUMSIMS }}

for sim in range(num_sims):
    bond_file = f"reax_run_{sim}.bonds"
    traj_file = f"reax_run_{sim}.lammpstrj"

    atom_lib = {{ NAME_TO_TYPE }}
    masses = {{ ATOM_MASSES }}

    sampler = prx.sample.Sample(atom_lib, masses, traj_file, bond_file, start_end_nthframe=(0,-1,1))
    atoms = [
        # {"atom": "a"},
        # {"atom": "b"},
        # {"atom": "c"},
        # {"atom": "a", "bonds": ["b", "b", "c"]},
    ]
    bonds = [
        # {"bond": "a-b", "bonds_A": ["b", "b", "b"], "bonds_B": ["c"]},
    ]
    pairs = [
        # ({"atom": "a", "bonds": ["b"]}, {"atom": "c", "bonds": ["d"]}),
    ]

    sampler.add_molecule_structure_sampling(f"run_{sim}_molecule_structure")

    sampler.add_charge_sampling(f"run_{sim}_charge_hist", atoms=atoms)

    sampler.add_bond_length_sampling(f"run_{sim}_bond_length", bonds=bonds, dimension="Bond Length", num_bins=200)

    sampler.add_angle_sampling(f"run_{sim}_angle_all", atoms=atoms, num_bins=180, angle="all")
    # sampler.add_angle_sampling(f"run_{sim}_angle_b-a-c", atoms=atoms, num_bins=90, angle="b-a-c")

    sampler.add_density_sampling(f"run_{sim}_density_time", atoms=atoms, dimension="Time")
    sampler.add_density_sampling(f"run_{sim}_density_cart_1d_z", atoms=atoms, dimension="Cartesian1D", num_bins=800)
    sampler.add_density_sampling(f"run_{sim}_density_cart_2d_xy", atoms=atoms, dimension="Cartesian2D", num_bins=200, direction="xy")
    sampler.add_density_sampling(f"run_{sim}_density_cart_1d_z_cond_ch", atoms=atoms, dimension="Cartesian1D", num_bins=200, direction="z", conditions={"Charge": (-0.7, -0.3)})
    sampler.add_density_sampling(f"run_{sim}_density_cart_2d_xy_cond_ang", atoms=atoms, dimension="Cartesian2D", num_bins=200, direction="xy", conditions={"Angle": (75, 125)})

    sampler.add_bond_density_sampling(f"run_{sim}_bond_density_cart_1d", bonds, "Cartesian1D", num_bins=800)

    sampler.add_rdf_sampling(f"run_{sim}_rdf", pairs, r_max=10.0, num_bins=200

    sampler.sample(is_parallel=True)
