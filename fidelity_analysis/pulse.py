from abc import ABC
import numpy as np
import skrf as rf
from matplotlib import pyplot as plt
from scipy.signal import czt, CZT
from scipy.interpolate import interp1d

from fidelity_analysis.utils import UnitConverter

USE_FFT = False
from mpl_toolkits.axes_grid1.inset_locator import inset_axes





class Pulse:
    def __init__(self,
                 pulse_samples_number: int,
                 name: str,
                 t_signal: np.ndarray = None,
                 t_signal_times: np.ndarray = None,
                 f_signal: np.ndarray = None,
                 f_signal_frequencies: np.ndarray = None,
                 frequencies_edges: tuple = None
                 ):
        if not (t_signal is None and t_signal_times is None and f_signal is None and f_signal_frequencies is None):
            if t_signal is None or t_signal_times is None or f_signal is None or f_signal_frequencies is None:
                raise ValueError("To initialize with predefined signals, all four arrays "
                                 "(t_signal, t_signal_times, f_signal, f_signal_frequencies) "
                                 "must be provided.")
            if not (len(t_signal) == len(t_signal_times) == len(f_signal) == len(f_signal_frequencies)):
                raise ValueError("All provided signal arrays must have the same length.")

            self.pulse_samples_number = len(t_signal)
            self.t_signal = t_signal
            self.t_signal_times = t_signal_times
            self.f_signal = f_signal
            self.f_signal_frequencies = f_signal_frequencies

        else:
            self.pulse_samples_number = pulse_samples_number
            self.t_signal = self.t_signal_times = None
            self.f_signal = self.f_signal_frequencies = None

        self.name = name
        self.frequencies_edges = frequencies_edges

    def create_pulse(self):
        # This method is intended to be overridden by child classes that generate
        # their pulse signals based on parameters. For a Pulse initialized with
        # predefined data, it does nothing.
        pass

    @staticmethod
    def to_frequency_domain(time_signal: np.ndarray,
                            dt: float,
                            pulse_samples_number: int,
                            frequencies_edges: tuple
                            ) -> (np.ndarray, np.ndarray):

        if USE_FFT:
            yf = np.fft.fft(time_signal)
            xf = np.fft.fftfreq(pulse_samples_number, dt)
            f_signal = np.fft.fftshift(yf)
            f_signal_frequencies = np.fft.fftshift(xf)

        else:
            """
              Zoom into frequencies_edges with M = pulse_samples_number points via CZT
            """

            M, W, A = Pulse._MWA(dt=dt,
                                 pulse_samples_number=pulse_samples_number,
                                 frequencies_edges=frequencies_edges,
                                 )

            f_signal = czt(time_signal, m=M, w=W, a=A)
            f_signal_frequencies = np.linspace(frequencies_edges[0], frequencies_edges[1], M, endpoint=False)

        return f_signal, f_signal_frequencies

    @staticmethod
    def to_time_domain(frequency_signal: np.ndarray,
                       dt: float,
                       pulse_samples_number: int,
                       frequencies_edges: tuple
                       ) -> np.ndarray:
        if USE_FFT:
            # Inverse shift the frequency domain signal
            ifft_shifted_data = np.fft.ifftshift(frequency_signal)
            # Perform inverse FFT
            t_signal = np.fft.ifft(ifft_shifted_data)
        else:
            """
            Inverse CZT over the same zoom window;
            """

            M, W, A = Pulse._MWA(dt=dt,
                                 pulse_samples_number=pulse_samples_number,
                                 frequencies_edges=frequencies_edges,
                                 )

            N_out = pulse_samples_number
            W_inv = 1 / W
            A_inv = 1 / A

            # The number of points in the frequency signal is M
            num_freq_points = len(frequency_signal)

            y_conj = np.conj(frequency_signal)

            # Use a flexible CZT that takes M points in and produces N points out
            t_signal_conj_scaled = czt(y_conj, m=N_out, w=W_inv, a=A_inv)

            N_time_points = pulse_samples_number
            t_signal = np.conj(t_signal_conj_scaled) / N_time_points
        return t_signal

    @staticmethod
    def _MWA(dt: float,
             pulse_samples_number: int,
             frequencies_edges: tuple
             ):
        """
          Helper function for to_frequency_domain and to_time_domain functions
        """
        fs = 1.0 / dt

        f1 = frequencies_edges[0]
        f2 = frequencies_edges[1]
        M = pulse_samples_number

        # same W, A as forward, but inverted
        W = np.exp(-2j * np.pi * (f2 - f1) / (fs * M))
        A = np.exp(2j * np.pi * f1 / fs)

        return M, W, A

    @staticmethod
    def get_s_parameter(pulse_freqs: np.ndarray, ntw: rf.Network, param_index: tuple):
        # Extract original S-parameter and frequencies from the network
        s_param_original = ntw.s[:, param_index[0], param_index[1]]
        frequencies_original = ntw.f

        # Interpolate S-parameter (real and imaginary parts separately)
        interp_s_param_real = interp1d(frequencies_original, np.real(s_param_original),
                                       kind='linear', bounds_error=False, fill_value=0)
        interp_s_param_imag = interp1d(frequencies_original, np.imag(s_param_original),
                                       kind='linear', bounds_error=False, fill_value=0)

        s_param_interpolated_real = interp_s_param_real(pulse_freqs)
        s_param_interpolated_imag = interp_s_param_imag(pulse_freqs)

        s_param_processed = s_param_interpolated_real + 1j * s_param_interpolated_imag

        return s_param_processed

    def plot_pulse(self, plot_t_edges: tuple = None, plot_f_edges: tuple = None, fill_t_area: tuple = None, ):
        if self.t_signal is None or self.t_signal_times is None:
            print("Cannot plot time domain pulse: t_signal or t_signal_times is not populated.")
            return
        if self.f_signal is None or self.f_signal_frequencies is None:
            print("Cannot plot frequency domain pulse: f_signal or f_signal_frequencies is not populated.")
            return

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(self.t_signal_times, np.real(self.t_signal))
        ax[0].set_title('Time Domain')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(True)
        ax[0].tick_params(axis='x', labelrotation=45)
        if fill_t_area is not None:
            ax[0].axvspan(fill_t_area[0], fill_t_area[1], alpha=0.5, color='red')

        ax[1].plot(self.f_signal_frequencies, np.abs(self.f_signal))
        ax[1].set_title('Magnitude Spectrum')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Magnitude')
        ax[1].grid(True)

        def add_zoom_inset(outer_ax, x_data, y_data, xlim, location='upper right', zoom_size=(0.4, 0.4)):

            inset_ax = inset_axes(outer_ax, width=f"{zoom_size[0] * 100}%", height=f"{zoom_size[1] * 100}%", loc=location, borderpad=1.3)
            inset_ax.plot(x_data, y_data)
            inset_ax.set_xlim(xlim)
            inset_ax.grid(False)
            inset_ax.tick_params(labelsize=8)
            inset_ax.xaxis.offsetText.set_fontsize(8)
            inset_ax.yaxis.offsetText.set_fontsize(8)

            return inset_ax

        if plot_f_edges is not None:
            add_zoom_inset(
                outer_ax=ax[1],
                x_data=self.f_signal_frequencies,
                y_data=np.real(self.f_signal),
                xlim=plot_f_edges,
                location='upper right'
            )
        if plot_t_edges is not None:
            add_zoom_inset(
                outer_ax=ax[0],
                x_data=self.t_signal_times,
                y_data=np.real(self.t_signal),
                xlim=plot_t_edges,
                location='upper right'
            )

        plt.suptitle(self.name, fontsize=14)
        plt.tight_layout()
        plt.show()


class ReadoutPulse(Pulse, ABC):
    def __init__(self, pulse_duration: float, pulse_samples_number: int, name: str, frequencies_edges: tuple,
                 carrier_frequency: float, pulse_amplitude: float):
        super().__init__(pulse_samples_number=pulse_samples_number, name=name, frequencies_edges=frequencies_edges)
        self.pulse_duration = pulse_duration
        self.carrier_frequency = carrier_frequency
        self.pulse_amplitude = pulse_amplitude


class RectangularReadoutPulse(ReadoutPulse):
    def __init__(self,
                 carrier_frequency: float,
                 pulse_duration: float,
                 pulse_power_dbm: float,
                 total_signal_time: float,
                 pulse_samples_number: int = 2 ** 23,
                 name: str = "Rectangular Readout Pulse",
                 ):
        self.carrier_frequency = carrier_frequency
        self.frequencies_edges = (self.carrier_frequency - 200 * 10 ** 6, self.carrier_frequency + 200 * 10 ** 6)
        self.pulse_duration = pulse_duration
        self.pulse_amplitude = UnitConverter().dbm_to_amplitude(pulse_power_dbm)
        self.total_signal_time = total_signal_time
        self.pulse_start_time = (self.total_signal_time - self.pulse_duration) / 2  # Store for plotting

        super().__init__(pulse_duration=pulse_duration,
                         pulse_samples_number=pulse_samples_number,
                         name=name,
                         frequencies_edges=self.frequencies_edges,
                         carrier_frequency=carrier_frequency,
                         pulse_amplitude=self.pulse_amplitude,
                         )

        self.create_pulse()

    def create_pulse(self):
        dt = self.total_signal_time / self.pulse_samples_number
        t_signal_times = np.linspace(0, self.total_signal_time, self.pulse_samples_number, endpoint=False)

        pulse_start_time = (self.total_signal_time - self.pulse_duration) / 2
        t_signal_base = np.zeros(self.pulse_samples_number, dtype=complex)
        condition = (t_signal_times >= pulse_start_time) & \
                    (t_signal_times < pulse_start_time + self.pulse_duration)
        t_signal_base[condition] = self.pulse_amplitude

        self.t_signal = t_signal_base * np.exp(1j * self.carrier_frequency * 2 * np.pi * t_signal_times)
        self.t_signal_times = t_signal_times

        self.f_signal, self.f_signal_frequencies = self.to_frequency_domain(time_signal=self.t_signal,
                                                                            dt=dt,
                                                                            pulse_samples_number=self.pulse_samples_number,
                                                                            frequencies_edges=self.frequencies_edges
                                                                            )

    def plot_pulse(self, plot_t_edges: tuple = None, plot_f_edges: tuple = None, fill_t_area: tuple = None, ):
        plot_f_edges = (self.carrier_frequency * 0.999, self.carrier_frequency * 1.001)
        plot_t_edges = (self.pulse_start_time, self.pulse_start_time + self.pulse_duration * 0.001)
        super().plot_pulse(plot_t_edges=plot_t_edges, plot_f_edges=plot_f_edges, fill_t_area=fill_t_area)


class ReflectedPulse(Pulse):
    def __init__(self, original_pulse: Pulse, ntw: rf.Network, name: str = "Reflected Pulse"):
        s11_at_pulse_freq = self.get_s_parameter(original_pulse.f_signal_frequencies, ntw, param_index=(0, 0))

        reflected_f_signal = original_pulse.f_signal * s11_at_pulse_freq

        dt = original_pulse.t_signal_times[1] - original_pulse.t_signal_times[0]

        reflected_t_signal = self.to_time_domain(frequency_signal=reflected_f_signal,
                                                 dt=dt,
                                                 pulse_samples_number=original_pulse.pulse_samples_number,
                                                 frequencies_edges=original_pulse.frequencies_edges
                                                 )

        super().__init__(
            t_signal=reflected_t_signal,
            t_signal_times=original_pulse.t_signal_times,
            f_signal=reflected_f_signal,
            f_signal_frequencies=original_pulse.f_signal_frequencies,
            pulse_samples_number=original_pulse.pulse_samples_number,
            name=name,
            frequencies_edges=original_pulse.frequencies_edges
        )

    def create_pulse(self):
        # Signals are already set in __init__
        pass


class TransitedPulse(Pulse):
    def __init__(self, original_pulse: Pulse, ntw: rf.Network, name: str = "Transited Pulse"):
        s21_at_pulse_freq = self.get_s_parameter(original_pulse.f_signal_frequencies, ntw, param_index=(1, 0))

        transmitted_f_signal = original_pulse.f_signal * s21_at_pulse_freq
        dt = original_pulse.t_signal_times[1] - original_pulse.t_signal_times[0]
        transmitted_t_signal = self.to_time_domain(frequency_signal=transmitted_f_signal,
                                                   dt=dt,
                                                   pulse_samples_number=original_pulse.pulse_samples_number,
                                                   frequencies_edges=original_pulse.frequencies_edges
                                                   )

        super().__init__(
            t_signal=transmitted_t_signal,
            t_signal_times=original_pulse.t_signal_times,
            f_signal=transmitted_f_signal,
            f_signal_frequencies=original_pulse.f_signal_frequencies,
            pulse_samples_number=original_pulse.pulse_samples_number,
            name=name,
            frequencies_edges=original_pulse.frequencies_edges
        )

    def create_pulse(self):
        # Signals are already set in __init__
        pass
