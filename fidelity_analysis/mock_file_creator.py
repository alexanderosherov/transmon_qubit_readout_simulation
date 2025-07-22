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


def cut_spektrum(input_fp: str, output_fp: str, freq_min_ghz: float, freq_max_ghz: float):
    ntwk = rf.Network(input_fp)
    ntwk.f = ntwk.f / (10 ** 9) + 1
    ntwk = ntwk[f'{1 + freq_min_ghz}-{1 + freq_max_ghz}ghz']
    ntwk.f = ntwk.f / (10 ** 9) - 1
    output_dir = os.path.dirname(output_fp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    ntwk.write_touchstone(output_fp)


def dipSpreader(input_fp: str, output_fp: str, scale_factor: float):
    ntwk = rf.Network(input_fp)
    ntwk.f = ntwk.f / (10 ** 9) * scale_factor

    output_dir = os.path.dirname(output_fp)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    ntwk.write_touchstone(output_fp)


if __name__ == "__main__":
    IN_FILE = "template_0ghz_resonator_cut_wide.s2p"
    OUT_FILE = "template_0ghz_resonator_cut_wide_shift.s2p"

    current_path = os.path.abspath("")
    data_dir_path = os.path.join(current_path, "data")
    in_file_path = os.path.join(data_dir_path, IN_FILE)
    out_file_path = os.path.join(data_dir_path, OUT_FILE)

    # SHIFT = -7266900000/1e9
    #
    # shift_s2p_freq_axis_minimal(in_file_path, out_file_path, SHIFT)
    # hz2ghz(in_file_path, out_file_path)
    # cut_spektrum(in_file_path, out_file_path, freq_min_ghz=-0.01, freq_max_ghz=0.01)
    # dipSpreader(in_file_path, out_file_path, scale_factor=25)
    shift_s2p_freq_axis_minimal(in_file_path, out_file_path, 6.065852884)

    print("DONE")
