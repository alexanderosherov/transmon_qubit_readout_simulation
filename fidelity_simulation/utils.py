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
