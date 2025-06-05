import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import os
from scipy.interpolate import interp1d


def s11_s21_from_s2p_file(simulation_results_filename: str,
                          target_pulse_frequencies: np.ndarray = None,
                          plot: bool = False):
    current_path = os.path.abspath("")
    data_dir_path = os.path.join(current_path, "data")
    file_path = os.path.join(data_dir_path, simulation_results_filename)

    try:
        network = rf.Network(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None

    s11_original = network.s[:, 0, 0]
    s21_original = network.s[:, 1, 0]
    frequencies_original = network.f

    s11_processed = s11_original
    s21_processed = s21_original
    frequencies_processed = frequencies_original

    min_orig_freq = np.min(frequencies_original)
    max_orig_freq = np.max(frequencies_original)

    if target_pulse_frequencies is not None:
        print(f"Interpolating S-parameters to {len(target_pulse_frequencies)} target frequencies.")

        min_target_freq = np.min(target_pulse_frequencies)
        max_target_freq = np.max(target_pulse_frequencies)

        if min_target_freq < min_orig_freq or max_target_freq > max_orig_freq:
            print(f"Warning: Target frequencies [{min_target_freq:.2e} Hz, {max_target_freq:.2e} Hz] "
                  f"are outside the original S-parameter frequency range "
                  f"[{min_orig_freq:.2e} Hz, {max_orig_freq:.2e} Hz]. "
                  "Interpolation might extrapolate.")

        # Interpolate S11
        interp_s11_real = interp1d(frequencies_original, np.real(s11_original), kind='linear',
                                   bounds_error=False,
                                   fill_value=0)  # Can also use fill_value="extrapolate" for some scipy versions
        interp_s11_imag = interp1d(frequencies_original, np.imag(s11_original), kind='linear',
                                   bounds_error=False, fill_value=0)
        s11_interpolated_real = interp_s11_real(target_pulse_frequencies)
        s11_interpolated_imag = interp_s11_imag(target_pulse_frequencies)
        s11_processed = s11_interpolated_real + 1j * s11_interpolated_imag

        # Interpolate S21
        interp_s21_real = interp1d(frequencies_original, np.real(s21_original), kind='linear',
                                   bounds_error=False, fill_value=0)
        interp_s21_imag = interp1d(frequencies_original, np.imag(s21_original), kind='linear',
                                   bounds_error=False, fill_value=0)
        s21_interpolated_real = interp_s21_real(target_pulse_frequencies)
        s21_interpolated_imag = interp_s21_imag(target_pulse_frequencies)
        s21_processed = s21_interpolated_real + 1j * s21_interpolated_imag

        frequencies_processed = target_pulse_frequencies
        print("Interpolation complete.")

    plot_edges = (min_orig_freq * 0.99, max_orig_freq * 1.01)
    if plot:
        # Create a new figure for these plots
        _, ax = plt.subplots(3, 1, sharex=True)

        # Plot 1: Magnitude
        ax[0].plot(frequencies_processed, np.abs(s11_processed), label='$|S_{11}|$')
        ax[0].plot(frequencies_processed, np.abs(s21_processed), label='$|S_{21}|$')
        ax[0].set_ylabel('$|S_{xx}|$')
        ax[0].set_xlim(plot_edges)
        ax[0].grid(True)
        ax[0].legend()
        ax[0].set_title(f'S-Parameters from {simulation_results_filename}')

        # Plot 2: Imaginary Part
        ax[1].plot(frequencies_processed, np.imag(s11_processed), label='$Im(S_{11})$')
        ax[1].plot(frequencies_processed, np.imag(s21_processed), label='$Im(S_{21})$')
        ax[1].set_ylabel('$Im(S_{xx})$')
        ax[1].set_xlim(plot_edges)
        ax[1].grid(True)
        ax[1].legend()

        # Plot 3: Real Part
        ax[2].plot(frequencies_processed, np.real(s11_processed), label='$Re(S_{11})$')
        ax[2].plot(frequencies_processed, np.real(s21_processed), label='$Re(S_{21})$')
        ax[2].set_ylabel('$Re(S_{xx})$')  # Corrected label
        ax[2].set_xlabel('Frequency (Hz)')  # Add x-label to the last plot
        ax[2].set_xlim(plot_edges)
        ax[2].grid(True)
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    return frequencies_processed, s11_processed, s21_processed


def create_rectangle_pulse(f_resonator: float, samples_number: int, plot: bool = False, ):
    # --- Time Domain Parameters ---
    total_signal_time = 4 * 10 ** (-4)  # Total time duration of the signal (seconds)
    dt = total_signal_time / samples_number  # Time step (sampling interval)
    pulse_t_times = np.linspace(0, total_signal_time, samples_number, endpoint=False)  # Time vector

    # --- Rectangular Pulse Parameters ---
    pulse_width = 3.5 * 10 ** (-6)  # (seconds)
    pulse_amplitude = 1.0  # a.u.
    pulse_start_time = (total_signal_time - pulse_width) / 2  # (seconds)
    pulse_frequency = f_resonator

    # --- Create the Rectangular Pulse in Time Domain ---
    pulse_t_signal = np.zeros(samples_number)
    pulse_t_signal[
        (pulse_t_times >= pulse_start_time) & (pulse_t_times < pulse_start_time + pulse_width)] = pulse_amplitude
    pulse_t_signal = pulse_t_signal * np.exp(1j * pulse_frequency * 2 * np.pi * pulse_t_times)  # Phase = 0

    # --- To Frequency Domain ---
    yf = np.fft.fft(pulse_t_signal)
    xf = np.fft.fftfreq(samples_number, dt)

    pulse_f_signal = np.fft.fftshift(yf)
    pulse_f_frequencies = np.fft.fftshift(xf)

    # --- Plotting ---
    pulse_f_plot_edges = (pulse_frequency * 0.99, pulse_frequency * 1.01)
    pulse_t_plot_edges = (pulse_start_time * 0.99, pulse_start_time * 1.01 + pulse_width)
    if plot:
        plt.figure(figsize=(12, 8))

        fig, ax = plt.subplots(2, 1)

        # Plot 1: Time Domain
        ax[0].plot(pulse_t_times, np.real(pulse_t_signal))
        ax[0].set_title('Time Domain')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlim(pulse_t_plot_edges)
        ax[0].grid(True)
        ax[0].tick_params(axis='x', labelrotation=45)
        # Plot 2: Frequency Domain

        ax[1].plot(pulse_f_frequencies, np.abs(pulse_f_signal))
        ax[1].set_title('Magnitude Spectrum')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Magnitude')
        ax[1].set_xlim(pulse_f_plot_edges)
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    return pulse_f_frequencies, pulse_f_signal, pulse_t_signal, pulse_t_times, pulse_t_plot_edges


def readout_signal_after_interaction_with_resonator(s: np.ndarray, pulse: np.ndarray, frequencies: np.ndarray,
                                                    plot: bool = False):
    result = s * pulse
    if plot:
        max_amplitude_frequency = frequencies[np.argmax(result)]
        plt.plot(frequencies, np.abs(result))
        plt.title('Read out signal after interaction with resonator')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim((max_amplitude_frequency * 0.99, max_amplitude_frequency * 1.01))
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    return result
