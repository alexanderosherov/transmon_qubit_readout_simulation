import os

import skrf as rf
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


class UnitConverter:
    H_BAR = 1.054571817e-34

    @staticmethod
    def dbm_to_watts(power_dbm: float) -> float:
        return 10 ** (power_dbm / 10) * 1e-3

    @staticmethod
    def watts_to_db(power_watts: float) -> float:
        return np.log10(power_watts * 1e3) * 10

    def dbm_to_amplitude(self, power_dbm: float, impedance_ohms: float = 50.0) -> float:
        """
        Converts power in dBm to peak voltage amplitude.

        Args:
            power_dbm (float): Power in dBm.
            impedance_ohms (float): System impedance in Ohms (default is 50 Ohm).

        Returns:
            float: Peak voltage amplitude.
        """
        power_watts = self.dbm_to_watts(power_dbm)

        # Calculate RMS voltage: P = V_rms^2 / R  => V_rms = sqrt(P * R)
        voltage_rms = np.sqrt(power_watts * impedance_ohms)

        # Calculate peak voltage (assuming sinusoidal signal): V_peak = V_rms * sqrt(2)
        voltage_peak = voltage_rms * np.sqrt(2)

        return voltage_peak

    def dbm_to_photons(self, power_dbm, frequency_hz, kappa_total_rad_s):
        omega_r = 2 * np.pi * frequency_hz

        power_watts = self.dbm_to_watts(power_dbm)
        photons_number_in_resonator = power_watts / (self.H_BAR * omega_r * kappa_total_rad_s / 2)
        return photons_number_in_resonator

    def photons_to_dbm(self, photons_number_in_resonator, frequency_hz, kappa_total_rad_s):
        omega_r = 2 * np.pi * frequency_hz

        power_watts = photons_number_in_resonator * (self.H_BAR * omega_r * kappa_total_rad_s / 2)
        power_dbm = np.log10(power_watts / 1e-3) * 10

        return power_dbm


class S2pUtils:
    @staticmethod
    def plot_s2p(filepath):
        network = rf.Network(filepath)

        frequencies_ghz = network.f

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
        fig.suptitle(f'S-Parameters (Magnitude & Phase) for\n{filepath}', fontsize=16)

        ax_s11 = axes[0]
        ax_s21 = axes[1]

        try:
            s11_complex = network.s[:, 0, 0]
            ax_s11.plot(frequencies_ghz, 20 * np.log10(np.abs(s11_complex)), label='S11 Magnitude (dB)')
            ax_s11.set_ylabel('Magnitude (dB)')
            ax_s11.set_title('S11')
            ax_s11.grid(True)
            ax_s11_phase = ax_s11.twinx()
            ax_s11_phase.plot(frequencies_ghz, np.angle(s11_complex, deg=True), label='S11 Phase (deg)', c="C1")
            ax_s11_phase.set_ylabel('Phase (degrees)')
            lines, labels = ax_s11.get_legend_handles_labels()
            lines2, labels2 = ax_s11_phase.get_legend_handles_labels()
            ax_s11.legend(lines + lines2, labels + labels2, loc='best')
        except Exception as e:
            print(e)

        try:
            s21_complex = network.s[:, 1, 0]
            ax_s21.plot(frequencies_ghz, 20 * np.log10(np.abs(s21_complex)), label='S21 Magnitude (dB)')
            ax_s21.set_ylabel('Magnitude (dB)')
            ax_s21.set_title('S21')
            ax_s21.grid(True)
            ax_s21_phase = ax_s21.twinx()
            ax_s21_phase.plot(frequencies_ghz, np.angle(s21_complex, deg=True), label='S21 Phase (deg)', c="C1")
            ax_s21_phase.set_ylabel('Phase (degrees)')
            lines, labels = ax_s21.get_legend_handles_labels()
            lines2, labels2 = ax_s21_phase.get_legend_handles_labels()
            ax_s21.legend(lines + lines2, labels + labels2, loc='best')
        except Exception as e:
            print(e)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_resonator_S21(f_hz, Q_e, Q):
        def full_model(f, fr, Ql, Qc):
            term1 = Ql / Qc
            term2 = 1 + 2j * Ql * (f - fr) / fr
            return 1 - term1 / term2

        f_arr = np.linspace(f_hz * 0.999, f_hz * 1.001, 10000)
        S21 = full_model(f_arr, f_hz, Q, Q_e)
        freqs_ntw = rf.Frequency.from_f(f_arr, unit='Hz')

        s_params = np.zeros((len(f_arr), 2, 2), dtype=complex)
        s_params[:, 1, 0] = S21

        return rf.Network(frequency=freqs_ntw, s=s_params)


def setup_plotting(dpi=300):
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['axes.labelsize'] = 14

    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#669bbc', '#ee6055', '#60d394'])
    # plt.rcParams['axes.prop_cycle'] = cycler(color=[ '#354757', '#93342D',  '#D88848', ])
    plt.rcParams['axes.prop_cycle'] = cycler(color=['#2F4858', '#A02829', '#85BBD8', '#97A169', '#C8A0DC'])

# if "__main__" == __name__:
#     FILE = "data_00000_0_ghz.s2p"
#
#     current_path = os.path.abspath("")
#     data_dir_path = os.path.join(current_path, "data")
#     file_path = os.path.join(data_dir_path, FILE)
#     S2pUtils.plot_s2p(file_path)
