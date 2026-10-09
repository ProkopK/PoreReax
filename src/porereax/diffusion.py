"""
Module for sampling diffusion.

The module provides :class:`DiffusionSampler` for diffusion sampling .
"""

import numpy as np

from porereax.meta_sampler import (
    _NAME_OUT_PARAM,
    _SAMPLER_INIT_PARAMS,
    AtomSampler,
    _build_mol_dictionary,
    _validate_double_atoms,
)
from porereax.utils import Substitution


@Substitution(params=_SAMPLER_INIT_PARAMS, name_out=_NAME_OUT_PARAM)
class DiffusionSampler(AtomSampler):
    """
    Sampler class for diffusion.

    Parameters
    ----------

    """

    def __init__(

    ):



    if self._is_diffusion_vacf:
        inp_diff = inp.copy()
        inp_diff.update(self._diff_vacf_inp)
        inp_diff.pop("output")
        data_diff = output[0]["diffusion_vacf"]
        for out in output[1:]:
            for pore_id in output[0]["diffusion_vacf"].keys() if self._pore else [None]:
                if pore_id:
                    if pore_id[:5]!="shape":
                        continue
                    data_diff[pore_id]["density"] += out["diffusion_vacf"][pore_id]["density"]
                    data_diff[pore_id]["vacf_data"] += out["diffusion_vacf"][pore_id]["vacf_data"]
                else:
                    data_diff["density"] += out["diffusion_vacf"]["density"]
                    data_diff["vacf_data"] += out["diffusion_vacf"]["vacf_data"]

        # Save results in dictionary
        results = {system["sys"]: system["props"], "inp": inp_diff, "data": data_diff, "type": "diff_vacf"}




##################
# VACF Diffusion #
##################
def init_diffusion_vacf(self, link_out: str, len_correration=2e-11, new_time_origin=2e-13, sample_step=20, len_frame=1e-15, bin_num=32, direction=2, sample_each_residue=False):
    """Enable diffusion sampling routine with the local VACF Alogrithm.
    
    This function samples the local velocity autocorrelation function for
    the diffusion calculation in all three directions. The local diffusion 
    coefficient in a direction :math:`\\alpha` is calculated by

    .. math::
        D_{\\alpha,l} = \\frac{1}{N_l}\\sum_{i=1}^{N}\\int_0^{\\infty} \\langle v_{\\alpha,i,l}(0)v_{\\alpha,i}(t)\\rangle dt

    with :math:`N_l` as the average number of molecules in the local region, 
    :math:`v_{\\alpha,i}` as the velocity of molecule :math:`i` in direction 
    :math:`\\alpha` and :math:`v_{\\alpha,i,l}(t)` as :math:`v_{\\alpha,i}(t)S_{q,i}(t)`
    with :math:`S_{q,i}` as a swich function that is 1 if the molecule
    :math:`i` is in the local region and 0 otherwise. 
    
    The local region is defined by a binning of the simulation box in the 
    chosen direction, if the direction is **0** (x-axis), **1** (y-axis), or
    **2** (z-axis). If the system is a pore system, consisting of pores all
    the same shape, sharing the same axis, the direction can be chosen as 
    "radial" to bin the system in bins according to the distance to the 
    center of the pore. The diffusion is then calculated in coordinates
    corresponding to the pore shape. For cylindrical pores, the diffusion is
    calculated in cylindrical coordinates (radial, tangential, axial), for
    slit pores in Cartesian coordinates (x, y, z).

    The VACF is sampled for a correlation time and choosing a new time origin
    after a certain time. 

    The method was described by Hunter et al. in the Journal of Chemical Physics. Theory Comput. , 18(12), pp.3357–3363.
    
    Parameters
    ----------
    link_out : string
        Link to obj data file
    len_correration : float, optional
        Length of the correlation time in seconds
    new_time_origin : float, optional
        Time between new time origins in seconds
    sample_step : integer, optional
        Step size of the sampling in your simulation in frames
    len_frame : float, optional
        Length of a frame in your simulation in seconds
    bin_num : integer, optional
        Number of bins to be used
    direction : integer or string, optional
        Direction of descretization of the simulation box (**0** (x-axis);
        **1** (y-axis); **2** (z-axis)) If the system is a pore system,
        the direction can also be "radial" to bin the system in radial bins
    sample_each_residue : bool, optional
        If True, the VACF is sampled for each residue separately, default is False
    """
    # Calculate correlation steps and new time origin steps
    corr_steps = int(round(len_correration/len_frame/sample_step))
    new_time_origin_steps = int(round(new_time_origin/len_frame/sample_step))
    if corr_steps < 1 or new_time_origin_steps < 1:
        print("VACF needs a correlation time longer than one frame. Please adjust len_correration and/or new_time_origin.")
        return
    if 2*corr_steps-1 >= self._num_frame:
        print("VACF correlation time is too long for the number of frames in the trajectory. Please reduce len_correration or use a longer trajectory.")
        return

    # Initialize
    if self._is_diffusion_bin:
        print("Binning and VACF-approaches cannot be run in parallel.")
        return
    if self._is_diffusion_mc:
        print("MC and VACF-approaches cannot be run in parallel.")
        return
    if self._is_numpy:
        print("VACF-approaches cannot be run in parallel with numpy approaches.")
        return
    if direction not in [0,1,2] and (direction=="radial" and not self._pore):
        print("Wrong directional input. Possible inputs are 0 (x-axis), 1 (y-axis), 2 (z-axis), and 'radial' (only for pore systems).")
        return
    if self._traj.split(".")[-1]!="trr":
        print("VACF needs a trajectory file with velocities. Please use a .trr file.")
        return
    if direction=="radial":
        # Check if all pores share the same cylindrical axis
        axes = []
        shape = []
        for pore_id in self._pore.keys():
            if pore_id[:5]=="shape":
                if self._pore_props[pore_id]["type"]=="CYLINDER":
                    axes.append(tuple(self._pore_props[pore_id]["focal"][0:2]))
                    shape.append(self._pore_props[pore_id]["type"])
                else:
                    print("Full radial diffusion only available for pore systems consisting of cylindrical pores.")
                    return
        if not all(axis == axes[0] for axis in axes):
            print("Radial diffusion only available for pore systems consisting of pores, all sharing the same cylindrical axis.")
            return
        if all(s == shape[0] for s in shape):
            if shape[0]=="CYLINDER":
                direction = "radial_cylindrical"
        else:
            print("Radial diffusion only available for pore systems consisting of pores of the same shape.")
            return

    # Enable routine
    self._is_diffusion_vacf = True

    # Calculate bins
    if direction=="radial_cylindrical":
        bins = self._bin_pore(bin_num)["bins"]
    else:
        bins = self._bin_box(bin_num, direction)["bins"]

    # Create input dictionary
    self._diff_vacf_inp = {"output": link_out,"bins": bins, "len_correration": len_correration,  
                            "new_time_origin": new_time_origin, "sample_step": sample_step, 
                            "len_frame": len_frame, "bin_num": bin_num, "direction": direction, 
                            "corr_steps": corr_steps, "new_time_origin_steps": new_time_origin_steps,
                            "num_res": self.num_res, "sample_each_residue": sample_each_residue}

def _diffusion_vacf_data(self) -> dict:
    """Create VACF diffusion data structure.

    Returns
    -------
    data : dictionary
        VACF diffusion data structure, containing vacf_data 
        (bin_num, corr_steps, num_residues, 3) and 
        density (bin_num, num_residues)
    """
    # Initialize
    bin_num = self._diff_vacf_inp["bin_num"]
    corr_steps = self._diff_vacf_inp["corr_steps"]
    sample_each_residue = self._diff_vacf_inp["sample_each_residue"]

    # Create dictionary
    data = {}

    # For pore systems
    if self._pore:
        for pore_id in self._pore.keys():
            if pore_id[:5]=="shape":
                data[pore_id] = {}
                if sample_each_residue:
                    data[pore_id]["vacf_data"] = np.zeros((bin_num, corr_steps, self.num_res, 3), float)
                    data[pore_id]["density"] = np.zeros((bin_num, self.num_res), int)
                else:
                    data[pore_id]["vacf_data"] = np.zeros((bin_num, corr_steps, 1, 3), float)
                    data[pore_id]["density"] = np.zeros((bin_num, 1), int)
    # For box systems
    else:
        # Initialize vacf and density data
        if sample_each_residue:
            data["vacf_data"] = np.zeros((bin_num, corr_steps, self.num_res, 3), float)
            data["density"] = np.zeros((bin_num, self.num_res), int)
        else:
            data["vacf_data"] = np.zeros((bin_num, corr_steps, 1, 3), float)
            data["density"] = np.zeros((bin_num, 1), int)

    return data

def _diffusion_vacf(self, data: dict, frame_id: int, pos_list: np.ndarray, pos_pointer: int, vel_list: np.ndarray, vel_pointer: int):
    """
    This function samples the local velocity autocorrelation function (VACF) for
    the diffusion calculation.

    It uses a given buffer of velocities for a correlation time to sample the VACF
    if the current frame is a new time origin. The VACF is sampled for each bin
    forward and backward in time.
    
    Parameters
    ----------
    data : dictionary
        Data dictionary containing bins for VACF diffusion
    frame_id : integer
        Current frame id
    pos_list : numpy.ndarray
        List of molecule positions of all buffered frames
        shape: (corr_steps, num_molecules, 3)
    pos_pointer : integer
        Pointer to the current frame in the position list
    vel_list : numpy.ndarray
        List of molecule velocities of all buffered frames
        shape: (2*corr_steps-1, num_molecules, 3)
    vel_pointer : integer
        Pointer to the current frame in the velocity list
    """
    bins = self._diff_vacf_inp["bins"]
    direction = self._diff_vacf_inp["direction"]
    corr_steps = self._diff_vacf_inp["corr_steps"]
    new_time_origin_steps = self._diff_vacf_inp["new_time_origin_steps"]
    sample_each_residue = self._diff_vacf_inp["sample_each_residue"]

    if direction=="radial_cylindrical":
        direction = 0  # radial direction in cylindrical coordinates

    if (frame_id-corr_steps)%new_time_origin_steps==0:
        pos = pos_list[pos_pointer, :, direction] # positions of v(0), shape: (num_molecules); Pos pointer is at t=0

        # Vel pointer is at frame_id-(2*corr_steps-1) -> shift by +corr_steps-1 is middle of buffer which is t=0; mask is for t=0
        vel_pointer = (vel_pointer + corr_steps - 1) % vel_list.shape[0]  # Adjust pointer to t=0
        
        # Create forward and backward indices for correlation steps
        forward_index = (np.arange(corr_steps) + vel_pointer) % vel_list.shape[0]  # Indices for backward velocities
        backward_index = (np.arange(corr_steps)[::-1] + vel_pointer + corr_steps) % vel_list.shape[0]  # Indices for forward velocities

        # Loop over all pores or just the box
        for pore_id in self._pore.keys() if self._pore else [None]:
            # For pore systems
            if pore_id:
                if pore_id[:5]!="shape":
                    continue
                data_pore = data[pore_id]
                bin_pore = bins[pore_id]
                res = self._pore_props["box"]["res"]
                z_min = res + self._pore_props[pore_id]["focal"][2]-self._pore_props[pore_id]["length"]/2+self._entry
                z_max = res + self._pore_props[pore_id]["focal"][2]+self._pore_props[pore_id]["length"]/2-self._entry
                # Filter all molecules that are inside the pore in z-direction
                pore_mask = (z_min < pos_list[pos_pointer, :, 2]) & (pos_list[pos_pointer, :, 2] < z_max)
                in_wall_mask = (pos > (self._pore_props[pore_id]["diam"]*1.01)/2)
            # For box systems (all molecules are in the "pore" and not in the wall)
            else:
                data_pore = data
                bin_pore = bins
                pore_mask = np.ones(pos.shape, dtype=bool)
                in_wall_mask = np.zeros(pos.shape, dtype=bool)

            com_bins = np.digitize(pos, bin_pore) - 1  # bin indices for each molecule
            # If there are molecules that should be in a bin, print a warning
            if np.any(com_bins[pore_mask] >= len(bin_pore) - 1) or np.any(com_bins[pore_mask] < 0):
                print("WARNING: Some molecules are outside the defined bins at frame", frame_id, "Number of molecules outside bins:", np.sum((com_bins[pore_mask] >= len(bin_pore) - 1) | (com_bins[pore_mask] < 0)))
                print("WARNING: Position min/max:", np.min(pos[pore_mask]), np.max(pos[pore_mask]))

            for bin_id in range(len(bin_pore) - 1):
                bin_mask = (com_bins == bin_id)

                # All molecules to consider are in a bin, inside the pore, and not in the wall
                mask = bin_mask & pore_mask & ~in_wall_mask
                if not np.any(mask):
                    print("WARNING: No molecules found in bin", bin_id, "at frame", frame_id)
                    continue
                # Get filtered velocities for molecules in this bin (v_l(0) part in the equation)
                vel0 = vel_list[vel_pointer, mask, :]  # shape: (num_mol_in_bin, 3)
                # Build the velocity forward time series for these molecules (v(t) part in the equation)
                forward_velos = vel_list[forward_index][:, mask, :]  # shape: (corr_steps, num_mol_in_bin, 3); It is not the same as vel_list[forward_index, :, mask]
                # Build the velocity backward time series for these molecules (v(t) part in the equation)
                backward_velos = vel_list[backward_index][:, mask, :]  # shape: (corr_steps, num_mol_in_bin, 3)
                # Concatenate forward and backward
                vacf = (vel0[np.newaxis, :, :] * forward_velos + vel0[np.newaxis, :, :] * backward_velos) / 2 # shape: (corr_steps, num_mol_in_bin, 3)

                if sample_each_residue:
                    res_indices = np.where(mask)[0]
                    data_pore["vacf_data"][bin_id, :, :, :][:, res_indices, :] += vacf # It is not the same as [bin_id, :, res_indices, :]
                    data_pore["density"][bin_id, res_indices] += 1
                else:
                    data_pore["vacf_data"][bin_id, :, 0, :] += np.sum(vacf, axis=1)
                    data_pore["density"][bin_id, 0] += np.sum(mask)
