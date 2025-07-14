import skrf as rf
import os


def shift_s2p_freq_axis_minimal(input_fp: str, output_fp: str, freq_shift_Ghz: float):
    ntwk = rf.Network(input_fp)
    ntwk.f = ntwk.f / (10 ** 9) + freq_shift_Ghz

    output_dir = os.path.dirname(output_fp)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    ntwk.write_touchstone(output_fp)


if __name__ == "__main__":
    IN_FILE = "hfss_resonance_simulator_Resonator_7252612e-6GHz_2_ports_v12.s2p"
    OUT_FILE = "hfss_resonance_simulator_Resonator_7252612e-6GHz_2_ports_v12_shifted_mock_156_kHz.s2p"

    current_path = os.path.abspath("")
    data_dir_path = os.path.join(current_path, "data")
    in_file_path = os.path.join(data_dir_path, IN_FILE)
    out_file_path = os.path.join(data_dir_path, OUT_FILE)

    SHIFT = 156 * 10 ** (-6)

    shift_s2p_freq_axis_minimal(in_file_path, out_file_path, SHIFT)
    print("DONE")
