import skrf as rf
import os


def shift_s2p_freq_axis_minimal(input_fp: str, output_fp: str, freq_shift_ghz: float):
    ntwk = rf.Network(input_fp)
    ntwk.f = ntwk.f / (10 ** 9) + freq_shift_ghz

    output_dir = os.path.dirname(output_fp)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    ntwk.write_touchstone(output_fp)


def hz2ghz(input_fp: str, output_fp: str):
    ntwk = rf.Network(input_fp)
    ntwk.f = ntwk.f / (10 ** 9)

    output_dir = os.path.dirname(output_fp)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    ntwk.write_touchstone(output_fp)


if __name__ == "__main__":
    IN_FILE = "data_00000_0.s2p"
    OUT_FILE = "data_00000_0_ghz.s2p"

    current_path = os.path.abspath("")
    data_dir_path = os.path.join(current_path, "data")
    in_file_path = os.path.join(data_dir_path, IN_FILE)
    out_file_path = os.path.join(data_dir_path, OUT_FILE)

    # SHIFT = -7266900000/1e9
    #
    # shift_s2p_freq_axis_minimal(in_file_path, out_file_path, SHIFT)
    hz2ghz(in_file_path, out_file_path)
    print("DONE")
