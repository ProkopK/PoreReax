import porereax as prx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')


# TODO replace a, b, c, ... with your atom names from atom_lib

num_sims = {{ NUMSIMS }}

for sim in range(1, num_sims + 1):
    bond_file = f"reax_run_{sim}"
    traj_file = f"reax_run_{sim}"

    atom_lib = {{ NAME_TO_TYPE }}
    masses = {{ ATOM_MASSES }}

    sampler = prx.sample.Sample(atom_lib, masses, traj_file, bond_file, start_end_nthframe=(0,-1,1))
    atoms = [
        # {"atom": "a"},
        # {"atom": "b"},
        # {"atom": "c"},
        # {"atom": "a", "bonds": ["b", "c"]},
        ]
    bonds = [
        # {"bond": "a-b", "bonds_A": ["b", "b", "b"], "bonds_B": ["c"]},
        ]
    sampler.add_bond_structure_sampling(f"run_{sim}_bond_structure", "BondStructure")

    sampler.add_charge_sampling(f"run_{sim}_charge_hist", "Histogram", atoms=atoms)

    sampler.add_bond_length_sampling(f"run_{sim}_bond_length", "Histogram", bonds=bonds, num_bins=200)

    sampler.add_angle_sampling(f"run_{sim}_angle_all", "Histogram", atoms=atoms, num_bins=180, angle="all")
    # sampler.add_angle_sampling(f"run_{sim}_angle_b-a-b", "Histogram", atoms=atoms, num_bins=90, angle="b-a-b")

    sampler.add_density_sampling(f"run_{sim}_density_cart_1d_z", "Cartesian1D", atoms=atoms, num_bins=800)
    sampler.add_density_sampling(f"run_{sim}_density_cart_2d_xy", "Cartesian2D", atoms=atoms, num_bins=200, direction="xy")
    sampler.add_density_sampling(f"run_{sim}_density_cart_1d_z_cond_ch", "Cartesian1D", atoms=atoms, num_bins=200, direction="z", conditions={"Charge": (-0.7, -0.3)})
    sampler.add_density_sampling(f"run_{sim}_density_cart_2d_xy_cond_ang", "Cartesian2D", atoms=atoms, num_bins=200, direction="xy", conditions={"Angle": (75, 125)})

    sampler.add_density_sampling(f"run_{sim}_density_time", "Time", atoms=atoms)

    sampler.sample(is_parallel=True)
