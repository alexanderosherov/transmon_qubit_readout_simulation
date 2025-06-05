import numpy as np
import skrf as rf
from joblib import Parallel, delayed, cpu_count
from matplotlib import pyplot as plt
from tqdm import tqdm

from pulse import Pulse, TransitedPulse, ReflectedPulse, ReadoutPulse

"""
Fidelity simulation based on the paper

Wong, Hiu Yung, Prabjot Dhillon, Kristin M. Beck, and Yaniv J. Rosen. 2023. 
“A Simulation Methodology for Superconducting Qubit Readout Fidelity.” 
Solid-State Electronics 201 (March):108582. https://doi.org/10.1016/j.sse.2022.108582.
"""


class FidelitySimulation:
    def __init__(self,
                 readout_pulse: ReadoutPulse,
                 s_parameters_file_state_0: str,
                 s_parameters_file_state_1: str,
                 IQ_projection_frequency: int,
                 readout_type: str = "transition",
                 num_iterations: int = 50,
                 noise_parameters: dict = None
                 ):
        # readout_type can be 'transition' or 'reflection'

        self.s_parameters_file_state_0 = s_parameters_file_state_0
        self.s_parameters_file_state_1 = s_parameters_file_state_1
        self.readout_pulse = readout_pulse
        self.readout_type = readout_type
        self.IQ_projection_frequency = IQ_projection_frequency
        self.num_iterations = num_iterations

        # Noise parameters based on the paper
        if noise_parameters is None:
            noise_parameters = {
                'quantum_noise': {
                    'type': 'quantum',
                    'T_ns': 0.5,  # K, from paper
                },
                'thermal_noise_room_temp': {
                    'type': 'thermal',
                    'T_eff': 1.5,  # K, from paper
                    'bandwidth': 6e9,  # Hz (6 GHz), from paper
                    'resistance': 50.0  # Ohms, common impedance
                },
                'thermal_noise_hemt': {
                    'type': 'thermal',
                    'T_eff': 54,  # K, from paper
                    'bandwidth': 6e9,  # Hz (6 GHz), from paper
                    'resistance': 50.0  # Ohms
                }
            }
        self.noise_parameters = noise_parameters

    def run(self):
        ntw_state_0 = rf.Network(self.s_parameters_file_state_0)
        ntw_state_1 = rf.Network(self.s_parameters_file_state_1)

        if self.readout_type == "transition":
            signal_state_0 = TransitedPulse(original_pulse=self.readout_pulse, ntw=ntw_state_0)
            signal_state_1 = TransitedPulse(original_pulse=self.readout_pulse, ntw=ntw_state_1)
        elif self.readout_type == "reflection":
            signal_state_0 = ReflectedPulse(original_pulse=self.readout_pulse, ntw=ntw_state_0)
            signal_state_1 = ReflectedPulse(original_pulse=self.readout_pulse, ntw=ntw_state_1)
        else:
            raise NotImplementedError

        I_state_0, Q_state_0 = self._IQ_projection_homodyne_demodulation(signal_from_system=signal_state_0)
        I_state_1, Q_state_1 = self._IQ_projection_homodyne_demodulation(signal_from_system=signal_state_1)

        plt.scatter(I_state_0, Q_state_0, label="|0>")
        plt.scatter(I_state_1, Q_state_1, label="|1>")
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title('IQ Projection for State 0 and State 1')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Noise independent of signal power
    def _create_noise(self, signal_from_system: Pulse) -> np.ndarray:
        k = 1.3806e-23  # Boltzmann constant (J/K)

        signal_length = len(signal_from_system.t_signal)

        total_noise = np.zeros(signal_length)

        for noise_name, params in self.noise_parameters.items():
            noise_type = params['type']
            R = params.get('resistance', 50.0)  # Use individual resistance if defined, else 50

            if noise_type == 'quantum':
                T_ns = params.get('T_ns')

                # Bandwidth for quantum noise is 1/tp (from paper)
                # pulse_duration is in seconds but needed in ns
                B_quantum = 1.0 / (self.readout_pulse.pulse_duration * 10 ** 9)

                P_N_quantum = k * T_ns * B_quantum
                sigma = np.sqrt(P_N_quantum * R)

            elif noise_type == 'thermal':
                T_eff = params.get('T_eff')
                bandwidth = params.get('bandwidth')

                sigma = np.sqrt(4 * k * T_eff * bandwidth * R)

            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            total_noise += np.random.normal(0, sigma, size=signal_length)

        return total_noise

    def _IQ_projection_homodyne_demodulation(self, signal_from_system: Pulse):
        pulse_start = int(np.argmax(self.readout_pulse.t_signal > 0))
        pulse_end = np.argmax(self.readout_pulse.t_signal[pulse_start:] == 0) + pulse_start

        dt = signal_from_system.t_signal_times[1] - signal_from_system.t_signal_times[0]
        T = (pulse_end - pulse_start) * dt

        # Helper function to be parallelized
        def _process_single_projection():
            noise = self._create_noise(signal_from_system=signal_from_system)

            s = signal_from_system.t_signal.real + noise
            s_I = s / 2
            s_Q = s / 2

            A_lo = 1
            y_I = A_lo / 2 * np.cos(self.IQ_projection_frequency * signal_from_system.t_signal_times)
            y_Q = -A_lo / 2 * np.sin(self.IQ_projection_frequency * signal_from_system.t_signal_times)

            I_pre_integration = s_I * y_I * dt
            Q_pre_integration = s_Q * y_Q * dt

            I_val = 1 / T * np.sum(I_pre_integration[pulse_start:pulse_end])
            Q_val = 1 / T * np.sum(Q_pre_integration[pulse_start:pulse_end])
            return I_val, Q_val

        # Parallelize the loop using joblib
        results = Parallel(n_jobs=-1)(
            delayed(_process_single_projection)()
            for _ in tqdm(range(self.num_iterations))
        )

        # Unpack the results
        I = [res[0] for res in results]
        Q = [res[1] for res in results]

        return I, Q
