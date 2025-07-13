from typing import Tuple, List

import numpy as np
import scqubits as scq
from scipy.optimize import minimize


class TransmonResonatorFitter:
    """
    A class to fit the parameters of a dispersively coupled transmon-resonator system
    to measured spectroscopic frequencies.
    """

    def __init__(self, f01_measured: float, f12_measured: float,
                 fr0_measured: float, fr1_measured: float, fr2_measured: float,
                 ncut_transmon: int = 100, truncated_dim_transmon: int = 10,
                 truncated_dim_resonator: int = 10, evals_count: int = 10):
        """
        Initializes the fitter with measured frequencies and scqubits parameters.

        Args:
            f01_measured (float): Measured |0>-|1> qubit transition frequency.
            f12_measured (float): Measured |1>-|2> qubit transition frequency.
            fr0_measured (float): Measured resonator frequency when qubit is in |0>.
            fr1_measured (float): Measured resonator frequency when qubit is in |1>.
            fr2_measured (float): Measured resonator frequency when qubit is in |2>.
            ncut_transmon (int): ncut parameter for scqubits Transmon object.
            truncated_dim_transmon (int): truncated_dim for scqubits Transmon object.
            truncated_dim_resonator (int): truncated_dim for scqubits Oscillator object.
            evals_count (int): Number of eigenvalues to calculate for the HilbertSpace.
        """
        self.f01_measured = f01_measured
        self.f12_measured = f12_measured
        self.fr0_measured = fr0_measured
        self.fr1_measured = fr1_measured
        self.fr2_measured = fr2_measured

        # scqubits parameters
        self.ncut_transmon = ncut_transmon
        self.truncated_dim_transmon = truncated_dim_transmon
        self.truncated_dim_resonator = truncated_dim_resonator
        self.evals_count = evals_count

        self.fitted_params = None
        self.fitted_frequencies = None

    def _model_frequencies(self, params: List[float]) -> np.ndarray:
        """
        Calculates the dressed qubit and resonator frequencies for given parameters
        using scqubits. This is an internal helper method.

        Args:
            params (list/array): [EJ, EC, f_r, g]

        Returns:
            tuple: (qubit_freq_01, qubit_freq_12, resonator_freq_q0, resonator_freq_q1, resonator_freq_q2)
                   and the HilbertSpace object.
        """
        EJ, EC, f_r, g_strength = params

        transmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.0,
                                ncut=self.ncut_transmon,
                                truncated_dim=self.truncated_dim_transmon)
        resonator = scq.Oscillator(E_osc=f_r, truncated_dim=self.truncated_dim_resonator)

        hs = scq.HilbertSpace([transmon, resonator])
        # Assume ng=0
        hs.add_interaction(
            g_strength=g_strength,
            op1=(transmon.n_operator, transmon),
            op2=(resonator.annihilation_operator() - resonator.creation_operator(), resonator),
            add_hc=True,
        )
        evals, _ = hs.eigensys(evals_count=self.evals_count)

        hs.generate_lookup()
        # Indices for qubit states with resonator in ground state: |q_state, r_state=0>
        q0_idx = hs.dressed_index((0, 0))
        q1_idx = hs.dressed_index((1, 0))
        q2_idx = hs.dressed_index((2, 0))

        # Indices for resonator states with qubit in specified state: |q_state, r_state=1>
        r0_idx = hs.dressed_index((0, 1))
        r1_idx = hs.dressed_index((1, 1))
        r2_idx = hs.dressed_index((2, 1))

        qubit_freq_01 = evals[q1_idx] - evals[q0_idx]
        qubit_freq_12 = evals[q2_idx] - evals[q1_idx]

        # Resonator frequencies conditioned on qubit state
        resonator_freq_qubit_g = evals[r0_idx] - evals[q0_idx]
        resonator_freq_qubit_e1 = evals[r1_idx] - evals[q1_idx]
        resonator_freq_qubit_e2 = evals[r2_idx] - evals[q2_idx]

        return np.array([qubit_freq_01, qubit_freq_12,
                         resonator_freq_qubit_g, resonator_freq_qubit_e1, resonator_freq_qubit_e2])

    def _cost_function(self, params: List[float]) -> float:
        """
        The objective function to minimize during fitting.
        This is an internal helper method.
        """
        model_freqs_array = self._model_frequencies(params)
        measured_freqs_array = np.array([self.f01_measured, self.f12_measured,
                                         self.fr0_measured, self.fr1_measured, self.fr2_measured])
        return np.sum((model_freqs_array - measured_freqs_array) ** 2)

    def _calculate_initial_guess(self) -> List[float]:
        """
        Calculates initial guesses for EJ, EC, f_r, and g based on measured frequencies.
        This is an internal helper method.
        """
        fr_pull_guess = (self.fr0_measured + self.fr1_measured) / 2
        chi_guess = (self.fr0_measured - self.fr1_measured) / 2
        alpha_guess = self.f12_measured - self.f01_measured
        delta_guess = np.abs(self.f01_measured - fr_pull_guess)

        ec_guess = -alpha_guess
        ej_guess = (self.f01_measured + ec_guess) ** 2 / (8 * ec_guess)

        g_arg = -(chi_guess * delta_guess) * (1 + delta_guess / alpha_guess)
        g_guess = np.sqrt(g_arg)

        return [ej_guess, ec_guess, fr_pull_guess, g_guess]

    def fit_parameters(self, method: str = 'Nelder-Mead') -> np.ndarray:
        """
        Performs the optimization to fit the EJ, EC, f_r, and g parameters.

        Args:
            method (str): The optimization method for scipy.optimize.minimize.

        Returns:
            array: The fitted parameters [EJ, EC, f_r, g].
        """
        initial_guess = self._calculate_initial_guess()
        print(f"Initial guess for [EJ, EC, fr, g]: {initial_guess}")

        result = minimize(self._cost_function, np.array(initial_guess), method=method)

        self.fitted_params = result.x
        self.fitted_frequencies = self._model_frequencies(self.fitted_params)

        if not result.success:
            print(f"Warning: Fitting did not converge successfully. Message: {result.message}")

        return self.fitted_params

    def print_results(self):
        """Prints the fitted parameters and frequencies."""
        if self.fitted_params is None:
            print("Please run fit_parameters() first.")
            return

        print("\n--- Fitting Results ---")
        print(f"Fitted EJ: {self.fitted_params[0]:.4f} GHz")
        print(f"Fitted EC: {self.fitted_params[1]:.4f} GHz")
        print(f"Fitted Resonator Freq (bare, f_r): {self.fitted_params[2]:.4f} GHz")
        print(f"Fitted Coupling Strength (g): {self.fitted_params[3]:.4f} GHz")
        print("\nFitted Frequencies (from model with fitted parameters):")
        print(f"  Qubit f01: {self.fitted_frequencies[0]:.4f} GHz")
        print(f"  Qubit f12: {self.fitted_frequencies[1]:.4f} GHz")
        print(f"  Resonator freq @ Qubit |0>: {self.fitted_frequencies[2]:.4f} GHz")
        print(f"  Resonator freq @ Qubit |1>: {self.fitted_frequencies[3]:.4f} GHz")
        print(f"  Resonator freq @ Qubit |2>: {self.fitted_frequencies[4]:.4f} GHz")
        print("\nMeasured Frequencies (input data):")
        print(f"  Qubit f01: {self.f01_measured:.4f} GHz")
        print(f"  Qubit f12: {self.f12_measured:.4f} GHz")
        print(f"  Resonator freq @ Qubit |0>: {self.fr0_measured:.4f} GHz")
        print(f"  Resonator freq @ Qubit |1>: {self.fr1_measured:.4f} GHz")
        print(f"  Resonator freq @ Qubit |2>: {self.fr2_measured:.4f} GHz")
        print(f"\nTotal Squared Error: {self._cost_function(self.fitted_params):.2e}")


if __name__ == "__main__":
    # Measured frequencies (example data)
    f01_meas = 3.845965050
    f12_meas = 3.602264855

    fr0_meas = 6.065852884
    fr1_meas = 6.064931745
    fr2_meas = 6.064094073

    fitter = TransmonResonatorFitter(f01_meas, f12_meas, fr0_meas, fr1_meas, fr2_meas)

    fitted_parameters = fitter.fit_parameters()

    fitter.print_results()

    # Examples for later usage
    EJ_fitted, EC_fitted, fr_fitted, g_fitted = fitter.fitted_params
    fitted_frequencies_array = fitter.fitted_frequencies
