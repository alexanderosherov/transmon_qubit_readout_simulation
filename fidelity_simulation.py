import numpy as np
import skrf as rf
from joblib import Parallel, delayed, cpu_count
from matplotlib import pyplot as plt
from tqdm import tqdm

from pulse import Pulse, TransitedPulse, ReflectedPulse


class FidelitySimulation:
    def __init__(self,
                 readout_pulse: Pulse,
                 s_parameters_file_state_0: str,
                 s_parameters_file_state_1: str,
                 IQ_projection_frequency: int,
                 readout_type: str = "transition",
                 num_iterations: int = 50,
                 ):
        # readout_type can be 'transition' or 'reflection'

        self.s_parameters_file_state_0 = s_parameters_file_state_0
        self.s_parameters_file_state_1 = s_parameters_file_state_1
        self.readout_pulse = readout_pulse
        self.readout_type = readout_type
        self.IQ_projection_frequency = IQ_projection_frequency
        self.num_iterations = num_iterations

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


        plt.scatter(I_state_0, Q_state_0)
        plt.scatter(I_state_1, Q_state_1)
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title('IQ Projection for State 0 and State 1')
        plt.grid(True)
        plt.show() # Added plt.show() to display the plot


    @staticmethod
    def _create_noise(signal_from_system: Pulse):
        noise_amplitude = 3.5 * 10 ** (-4)
        return np.random.normal(0, 1, size=len(signal_from_system.t_signal)) * noise_amplitude

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
