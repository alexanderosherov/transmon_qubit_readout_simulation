import xarray as xr
import numpy as np
import skrf as rf
import copy
import os


def reformate_to_s2p(input_fp: str, output_fp: str, base_frequency: float):
    xr.set_options(keep_attrs=True)
    data_xr = xr.load_dataset(input_fp, engine="h5netcdf")

    output_fp_0 = output_fp.replace(".s2p", "_0.s2p")
    output_fp_1 = output_fp.replace(".s2p", "_1.s2p")
    output = (output_fp_0, output_fp_1)

    for i in range(2):
        f = copy.deepcopy(data_xr['data'].squeeze().frequency.data + base_frequency)
        signal = copy.deepcopy(data_xr['data'].squeeze().isel(iter=i).data)

        ntwk = rf.Network()
        ntwk.f = f * 1e-9
        s = np.zeros((len(signal), 2, 2), dtype=np.complex64)
        s[:, 1, 0] = signal
        ntwk.s = s

        output_dir = os.path.dirname(output[i])
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        ntwk.write_touchstone(output[i])


if __name__ == "__main__":
    IN_FILE = "data_00000.ncd"
    OUT_FILE = "data_00000.s2p"

    current_path = os.path.abspath("")
    data_dir_path = os.path.join(current_path, "data")
    in_file_path = os.path.join(data_dir_path, IN_FILE)
    out_file_path = os.path.join(data_dir_path, OUT_FILE)

    BASE_FREQUENCY = 6065802860

    reformate_to_s2p(in_file_path, out_file_path, BASE_FREQUENCY)
    print("DONE")
