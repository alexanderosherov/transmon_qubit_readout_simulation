import skrf as rf
import matplotlib.pyplot as plt
import numpy as np

class UnitConverter:
    @staticmethod
    def dbm_to_amplitude(power_dbm: float, impedance_ohms: float = 50.0) -> float:
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

class S2pUtils:
    @staticmethod
    def plot_s2p(filepath):
        network = rf.Network(filepath)

        frequencies_ghz = network.f

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
        fig.suptitle(f'S-Parameters (Magnitude & Phase) for\n{filepath}', fontsize=16)

        ax_s11 = axes[0]
        ax_s21 = axes[1]

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


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


