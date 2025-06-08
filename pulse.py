from abc import ABC

import numpy as np
import skrf as rf
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class Pulse:
    def __init__(self, pulse_samples_number: int,
                 t_signal: np.ndarray = None, t_signal_times: np.ndarray = None,
                 f_signal: np.ndarray = None, f_signal_frequencies: np.ndarray = None):
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
            self.f_signal_frequencies = None
            self.f_signal = None
            self.t_signal = None
            self.t_signal_times = None

    def create_pulse(self):
        # This method is intended to be overridden by child classes that generate
        # their pulse signals based on parameters. For a Pulse initialized with
        # predefined data, it does nothing.
        pass

    @staticmethod
    def to_frequency_domain(time_signal: np.ndarray, dt: float, pulse_samples_number: int):
        yf = np.fft.fft(time_signal)
        xf = np.fft.fftfreq(pulse_samples_number, dt)

        f_signal_shifted = np.fft.fftshift(yf)
        f_signal_frequencies_shifted = np.fft.fftshift(xf)
        return f_signal_shifted, f_signal_frequencies_shifted

    @staticmethod
    def to_time_domain(frequency_signal: np.ndarray) -> np.ndarray:
        # Inverse shift the frequency domain signal
        ifft_shifted_data = np.fft.ifftshift(frequency_signal)
        # Perform inverse FFT
        t_signal = np.fft.ifft(ifft_shifted_data)
        return t_signal

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

    def plot_pulse(self, plot_t_edges: tuple = None, plot_f_edges: tuple = None):
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
        ax[0].set_xlim(plot_t_edges)
        ax[0].grid(True)
        ax[0].tick_params(axis='x', labelrotation=45)

        ax[1].plot(self.f_signal_frequencies, np.abs(self.f_signal))
        ax[1].set_title('Magnitude Spectrum')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Magnitude')
        ax[1].set_xlim(plot_f_edges)
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()


class ReadoutPulse(Pulse, ABC):
    def __init__(self, pulse_duration: float, pulse_samples_number: int):
        super().__init__(pulse_samples_number=pulse_samples_number)
        self.pulse_duration = pulse_duration


class RectangleReadoutPulse(ReadoutPulse):
    def __init__(self, carrier_frequency: float, pulse_duration: float,
                 pulse_power_dbm: float, total_signal_time: float,
                 pulse_samples_number: int = 2 ** 23):
        super().__init__(pulse_duration=pulse_duration, pulse_samples_number=pulse_samples_number)
        self.carrier_frequency = carrier_frequency
        self.pulse_duration = pulse_duration
        self.pulse_amplitude = self._dbm_to_amplitude(pulse_power_dbm)
        self.total_signal_time = total_signal_time
        self.pulse_start_time = (self.total_signal_time - self.pulse_duration) / 2  # Store for plotting

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

        self.f_signal, self.f_signal_frequencies = self.to_frequency_domain(self.t_signal, dt,
                                                                            self.pulse_samples_number)

    def plot_pulse(self, plot_t_edges: tuple = None, plot_f_edges: tuple = None):
        plot_f_edges = (self.carrier_frequency * 0.999, self.carrier_frequency * 1.001)
        plot_t_edges = (self.pulse_start_time * 0.99, (self.pulse_start_time + self.pulse_duration) * 1.01)
        super().plot_pulse(plot_t_edges=plot_t_edges, plot_f_edges=plot_f_edges)

    @staticmethod
    def _dbm_to_amplitude(power_dbm: float, impedance_ohms: float = 50.0) -> float:
        """
        Converts power in dBm to peak voltage amplitude.

        Args:
            power_dbm (float): Power in dBm.
            impedance_ohms (float): System impedance in Ohms (default is 50 Ohm).

        Returns:
            float: Peak voltage amplitude.
        """
        # Convert power from dBm to Watts
        power_watts = 10 ** ((power_dbm - 30) / 10)

        # Calculate RMS voltage: P = V_rms^2 / R  => V_rms = sqrt(P * R)
        voltage_rms = np.sqrt(power_watts * impedance_ohms)

        # Calculate peak voltage (assuming sinusoidal signal): V_peak = V_rms * sqrt(2)
        voltage_peak = voltage_rms * np.sqrt(2)

        return voltage_peak


class ReflectedPulse(Pulse):
    def __init__(self, original_pulse: Pulse, ntw: rf.Network):
        s11_at_pulse_freq = self.get_s_parameter(original_pulse.f_signal_frequencies, ntw, param_index=(0, 0))

        reflected_f_signal = original_pulse.f_signal * s11_at_pulse_freq
        reflected_t_signal = self.to_time_domain(reflected_f_signal)

        super().__init__(
            t_signal=reflected_t_signal,
            t_signal_times=original_pulse.t_signal_times,
            f_signal=reflected_f_signal,
            f_signal_frequencies=original_pulse.f_signal_frequencies,
            pulse_samples_number=original_pulse.pulse_samples_number
        )

    def create_pulse(self):
        # Signals are already set in __init__
        pass


class TransitedPulse(Pulse):
    def __init__(self, original_pulse: Pulse, ntw: rf.Network):
        s21_at_pulse_freq = self.get_s_parameter(original_pulse.f_signal_frequencies, ntw, param_index=(1, 0))

        transmitted_f_signal = original_pulse.f_signal * s21_at_pulse_freq
        transmitted_t_signal = self.to_time_domain(transmitted_f_signal)

        super().__init__(
            t_signal=transmitted_t_signal,
            t_signal_times=original_pulse.t_signal_times,
            f_signal=transmitted_f_signal,
            f_signal_frequencies=original_pulse.f_signal_frequencies,
            pulse_samples_number=original_pulse.pulse_samples_number
        )

    def create_pulse(self):
        # Signals are already set in __init__
        pass
