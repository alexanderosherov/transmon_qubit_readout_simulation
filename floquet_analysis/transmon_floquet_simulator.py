import numpy as np
from qutip import Qobj, FloquetBasis
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Dict
from joblib import Parallel, delayed


class TransmonFloquetSimulator:
    """
    A class to simulate Floquet quasienergies and averaged transmon excitation
    for a driven transmon qubit.
    """

    def __init__(self, Ec: float,
                 EjEc: float,
                 N_charge_basis: int,
                 drive_frequency_wd: float,
                 g_strength: float,
                 n_r: np.ndarray,
                 ):
        """
        Initializes the simulator with transmon and drive parameters.

        Args:
            Ec: Charging energy of the transmon (e.g., in GHz).
            EjEc: Ratio EJ/EC for the transmon.
            N_charge_basis: Truncation dimension for the charge basis (basis states from -N to N).
            drive_frequency_wd: Angular frequency of the drive (e.g., in GHz).
            g_strength: Coupling strength 'g'.
            n_r: Array of mean photon numbers to simulate.

        """
        self.Ec = Ec
        self.EjEc = EjEc
        self.N_charge_basis = N_charge_basis
        self.w_d = drive_frequency_wd
        self.n_r = n_r
        self.T = 2 * np.pi / self.w_d  # Drive period

        self.g = g_strength

        # Store simulation results
        self.floquet_branches = None
        self.averaged_excitation = None

        self.H_t_bare = None
        self.bare_eigenenergies, self.bare_eigenstates = None, None

    def find_n_r_critical(self, branch_index: int,
                          ng: float,
                          plot: bool = True,
                          branches_to_plot: Tuple[int] = None,
                          plot_range: Tuple[int, int] = None,
                          ) -> int:
        """
        Args:
            branch_index:
            ng:
            plot:
            branches_to_plot:
            plot_range:
        Returns:
            n_r_critical: Critical number of photons in the resonator

        """
        self.H_t_bare = self._hamiltonian_t_shifted(ng)
        self.bare_eigenenergies, self.bare_eigenstates = self.H_t_bare.eigenstates()

        self._calculate_floquet_branches(show_progress=plot)
        self._calculate_averaged_transmon_excitation(show_progress=plot)

        if plot:
            if branches_to_plot is None:
                branches_to_plot = [branch_index]
            self._plot_floquet_results_for_the_latest_ng(plot_range, branches_to_plot)

        ac_index = self._find_first_avoided_crossing(branch_index)

        return ac_index

    def find_minmax_n_r_critical(self, branch_index: int, ng: np.ndarray, plot: bool = True):
        """
        Args:
            branch_index: Index of the Branch we use to define critical number n_r_critical
            ng: Offset charge for the transmon.
            plot: If true dependence between n_r_critical from ng will be plotted.

        Returns:
            n_r_critical: Minimal and maximal critical number of photons in the resonator
        """
        n_critical = []
        for ng_val in tqdm(ng, desc="Finding min and max of n_r_critical"):
            n_critical_for_this_ng = self.find_n_r_critical(branch_index, ng_val, plot=False)
            n_critical.append(n_critical_for_this_ng)

        # n_critical = Parallel(n_jobs=-1)(
        #     delayed(self.find_n_r_critical)(branch_index, ng_val, plot=False)
        #     for ng_val in tqdm(ng, desc="Finding min and max of n_r_critical")
        # )

        # Filter out None values in case _find_first_avoided_crossing can return None
        n_critical = [val for val in n_critical if val is not None]
        print("Critical number of photons in the resonator: {}".format(n_critical))

        if not n_critical:
            return None, None  # Or raise an error, depending on desired behavior

        if plot:
            self._plot_n_critical_dependence_from_ng(n_critical, ng)

        return min(n_critical), max(n_critical)

    def _hamiltonian_t_shifted(self, ng_val: float) -> Qobj:
        """
        Returns the bare transmon Hamiltonian (H_t), shifted by its ground state energy.
        """
        n_arr = np.arange(-self.N_charge_basis, self.N_charge_basis + 1)
        Ej = self.EjEc * self.Ec
        H0_np = np.diag(4 * self.Ec * (n_arr - ng_val) ** 2) - \
                0.5 * Ej * (np.diag(np.ones(2 * self.N_charge_basis), 1) +
                            np.diag(np.ones(2 * self.N_charge_basis), -1))
        H0 = Qobj(H0_np)

        es, _ = H0.eigenstates()
        return H0 - Qobj(es[0] * np.eye(2 * self.N_charge_basis + 1))

    def _hamiltonian_d(self, n_r: float) -> Qobj:
        """
        Returns the amplitude of the driven term (epsilon_t(t)).
        The full driven term is H_driven = hammiltonian_d * cos(w_d * t).
        """
        n_arr = np.arange(-self.N_charge_basis, self.N_charge_basis + 1)
        sigma_t = 2 * self.g * np.sqrt(n_r)
        H1_np = sigma_t * np.diag(n_arr)
        return Qobj(H1_np)

    def _get_floquet_basis(self, n_r: float) -> FloquetBasis:
        """Helper to get FloquetBasis object for a given n_r."""
        H_list: List[Union[Qobj, List[Union[Qobj, str]]]] = [
            self.H_t_bare,
            [self._hamiltonian_d(n_r), 'cos(w_d*t)']
        ]
        args: Dict[str, float] = {"w_d": self.w_d}

        return FloquetBasis(H_list, self.T, args, sort=True, options=dict(nsteps=10000))

    def _calculate_floquet_branches(self, show_progress: bool = False) -> np.ndarray:
        """
        Calculates and sorts Floquet quasienergies for varying photon numbers.

        Returns:
            np.ndarray: Array of sorted Floquet quasienergies.
        """
        branches = []
        f_modes_list = []

        for n_r in tqdm(self.n_r, desc="Floquet Branches", disable=not show_progress):
            floquet_basis = self._get_floquet_basis(n_r)
            f_modes = floquet_basis.mode(0, False)  # Floquet modes at t=0
            f_energies = floquet_basis.e_quasi  # Quasienergies

            sorted_e = []
            sorted_m = []

            if n_r == 0:
                # For the first point (n_r = 0), sort by overlap with bare states
                # This assumes bare_eigenstates are sorted by energy for easy mapping
                for vec in self.bare_eigenstates:
                    overlaps = np.array([abs(vec.overlap(f_mode)) ** 2 for f_mode in f_modes])
                    idx_max_overlap = np.argmax(overlaps)
                    sorted_e.append(f_energies[idx_max_overlap])
                    sorted_m.append(f_modes[idx_max_overlap])
            else:
                # For subsequent points, sort by overlap with the previously sorted Floquet modes
                for last_f_mode in f_modes_list[-1]:
                    overlaps = np.array([abs(last_f_mode.overlap(f_mode)) ** 2 for f_mode in f_modes])
                    idx_max_overlap = np.argmax(overlaps)
                    sorted_e.append(f_energies[idx_max_overlap])
                    sorted_m.append(f_modes[idx_max_overlap])

            branches.append(sorted_e)
            f_modes_list.append(sorted_m)  # Store sorted modes for next iteration's overlap check

        self.floquet_branches = np.array(branches)
        return self.floquet_branches

    def _calculate_averaged_transmon_excitation(self, show_progress: bool = False) -> np.ndarray:
        """
        Calculates the period-averaged transmon excitation for each Floquet mode.

        This involves:
        1. Obtaining the Floquet modes at various time points within one period.
        2. Calculating the expectation value of the transmon number operator
           for each Floquet mode at each time point.
        3. Averaging these expectation values over one period (integration).

        Returns:
            np.ndarray: Array of averaged transmon excitation for each Floquet mode (branch).
                        Shape: (len(n_r_list), number_of_floquet_modes).
        """
        averaged_excitation_list_arr = []
        f_modes_list_for_sorting = []  # To ensure consistent sorting of branches

        for n_r in tqdm(self.n_r, desc="Avg Excitation", disable=not show_progress):
            floquet_basis = self._get_floquet_basis(n_r)

            # First, get Floquet modes at t=0 and sort them consistently
            f_modes_at_t0 = floquet_basis.mode(0, False)
            sorted_f_modes_t0 = []
            sorted_i = []
            if n_r == 0:
                for vec in self.bare_eigenstates:
                    overlaps = np.array([abs(vec.overlap(f_mode)) ** 2 for f_mode in f_modes_at_t0])
                    idx_max_overlap = np.argmax(overlaps)
                    sorted_f_modes_t0.append(f_modes_at_t0[idx_max_overlap])
                    sorted_i.append(idx_max_overlap)
            else:
                for last_f_mode in f_modes_list_for_sorting[-1]:
                    overlaps = np.array([abs(last_f_mode.overlap(f_mode)) ** 2 for f_mode in f_modes_at_t0])
                    idx_max_overlap = np.argmax(overlaps)
                    sorted_f_modes_t0.append(f_modes_at_t0[idx_max_overlap])
                    sorted_i.append(idx_max_overlap)

            f_modes_list_for_sorting.append(sorted_f_modes_t0)  # Store for next iteration's sorting

            excitation_per_mode = self._calculate_transmon_excitation(floquet_basis, sorted_i)

            averaged_excitation_list_arr.append(np.array(excitation_per_mode))

        self.averaged_excitation = np.array(averaged_excitation_list_arr)
        return self.averaged_excitation

    def _calculate_transmon_excitation(self, floquet_basis, sorted_i):
        # Number of time steps for averaging within one period.
        # This can be adjusted for accuracy vs. speed.
        num_time_points = 2 * self.N_charge_basis + 1  # Original was 2*N+1, which might be too few. 500 used previously.

        t_list = np.linspace(0, self.T, num_time_points)

        # Calculate the time step
        dt = t_list[1] - t_list[0]

        # Calculate Floquet modes at all-time points
        f_modes_all = []
        for t in t_list:
            modes = np.array(floquet_basis.mode(t))[sorted_i]
            f_modes_all.append(modes)

        # Calculate the weighted overlaps for all-time points
        overlaps_all = np.array([[[np.abs(vec.overlap(Qobj(f_mode_alpha))) ** 2 * psi_index
                                   for psi_index, vec in enumerate(self.bare_eigenstates)]
                                  for f_mode_alpha in f_modes_t]
                                 for f_modes_t in f_modes_all])
        # Calculate the total transmon excitation for each Floquet branch
        R_alpha = np.sum(overlaps_all, axis=2)

        # Integrate over time by summing along the time axis and normalize by the period T
        R_alpha = np.sum(R_alpha, axis=0) * dt / self.T
        # R_alpha_integrated = simpson(R_alpha, t_list, axis=0) * (1 / T)

        return R_alpha

    def _find_first_avoided_crossing(self, branch_index: int) -> Union[int, float]:
        """
        Finds the first "avoided crossing" proxy in a specific Floquet quasienergy branch.
        This is a simplified approach based on derivative sign changes.

        Args:
            branch_index: The index of the Floquet branch to analyze.

        Returns:
            int: The index in n_r_list where the first avoided crossing is detected,
                 or np.nan if no crossing is found.
        """
        if self.floquet_branches is None or branch_index >= self.floquet_branches.shape[1]:
            print(f"Error: Floquet branches not calculated or branch_index {branch_index} out of bounds.")
            return np.nan

        branch = self.floquet_branches[:, branch_index]
        deriv = np.diff(branch, axis=0)
        crossings = np.where(np.diff(np.sign(deriv)))[0]
        if len(crossings) > 0:
            return int(crossings[0] + 1)
        else:
            return np.nan

    def _plot_floquet_results_for_the_latest_ng(self, plot_range: Tuple[int, int],
                                                branches_to_plot: Tuple[int] = None) -> None:
        """
        Plots the Floquet quasienergies and averaged transmon excitation.

        Args:
            branches_to_plot: List of specific branch indices to highlight.
                              If None, will highlight the default first few.
            plot_range: Tuple (start, end) for the range of branches to plot.
                        All branches within this range will be plotted, with
                        'branches_to_plot' being highlighted.
        """

        if branches_to_plot is None:
            branches_to_plot = []

        num_branches = self.floquet_branches.shape[1]

        if plot_range is None:
            plot_range = [0, num_branches]

        fig, axs = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, dpi=300,
                                constrained_layout=True)

        colors = plt.cm.tab10.colors
        plot_colors = [colors[i % len(colors)] for i in range(len(branches_to_plot))]

        # --- Top Plot (Floquet Quasienergies) ---
        for i in range(plot_range[0], min(plot_range[1], num_branches)):
            if i not in branches_to_plot:
                axs[0].plot(self.n_r, self.floquet_branches[:, i] / self.w_d,
                            linewidth=0.8, color='grey', alpha=0.6, zorder=1)

        for idx, i_t in enumerate(branches_to_plot):
            axs[0].plot(self.n_r, self.floquet_branches[:, i_t] / self.w_d,
                        linewidth=2.5, label=f'$B_{i_t}$', color=plot_colors[idx], zorder=2)

        axs[0].legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=len(branches_to_plot),
                      fontsize=15, frameon=False, columnspacing=1.5, handlelength=1.5)
        axs[0].set_ylabel(r'$\epsilon_i / \omega_\text{d}$', fontsize=12)
        axs[0].set_ylim(-0.5, 0.5)

        # Axis styling for top plot
        axs[0].grid(True, linestyle='--', alpha=0.5, zorder=0)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].tick_params(axis='both', which='major', labelsize=10, length=4, width=1)

        # --- Bottom Plot (Averaged Transmon Excitation) ---
        for i in range(plot_range[0], min(plot_range[1], num_branches)):
            if i not in branches_to_plot:
                axs[1].plot(self.n_r, self.averaged_excitation[:, i],
                            linewidth=0.8, color='grey', alpha=0.6, zorder=1)

        for idx, i_t in enumerate(branches_to_plot):
            axs[1].plot(self.n_r, self.averaged_excitation[:, i_t],
                        linewidth=2.5, label=f'{i_t}_t', color=plot_colors[idx], zorder=2)

        axs[1].set_xlabel(r'$\bar{n}$', fontsize=12)
        axs[1].set_ylabel('$N_t$', fontsize=12)

        # Axis styling for bottom plot
        axs[1].grid(True, linestyle='--', alpha=0.5, zorder=0)  # Add a subtle grid
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].tick_params(axis='both', which='major', labelsize=10, length=4, width=1)

        plt.show()

    @staticmethod
    def _plot_n_critical_dependence_from_ng(n_critical: Union[np.ndarray, List], ng: np.ndarray) -> None:
        """
        Plots the dependece of critical photon number in ther esonator from ng

        Args:
            n_critical: List critical photon numbers to plot.
            ng: List of used ng values
        """

        fig, axs = plt.subplots(1, 1, figsize=(7, 5.5), sharex=True, dpi=300)

        axs.set_xlabel(r'$n_\text{g}$', fontsize=12)
        axs.set_ylabel(r'$\bar{n}_\text{c}$', fontsize=12)
        axs.grid(True, linestyle='--', alpha=0.5, zorder=0)
        print()
        axs.plot(ng, n_critical)

        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters for the Floquet simulation
    Ec_val = 1.328549866299301  # GHz
    EjEc_val = 46.40380378145438  # Ratio
    N_val = 10  # Charge basis truncation
    w_d_val = 38.109963742041856  # GHz (Drive frequency)

    # Define the range of resonator photon numbers
    n_r_list_sim = np.linspace(0, 150, 151)
    g_strength_sim = 0.48881501701797614  # Coupling strength in GHz

    # Initialize the simulator
    simulator = TransmonFloquetSimulator(Ec_val, EjEc_val, N_val, w_d_val, g_strength_sim, n_r_list_sim)

    # result = simulator.find_n_r_critical(branch_index=1, ng=0)
    result = simulator.find_minmax_n_r_critical(branch_index=1, ng=np.linspace(0, 0.5, 50))
    print("result:", result)
