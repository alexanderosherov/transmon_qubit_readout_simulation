import os
import numpy as np
import pya
from itertools import product
from tqdm import tqdm

from wmiklayout.config.layerconfig import default_layers
from wmiklayout.utils.io_functions import export_layout
from wmiklayout.utils.library_functions import load_libraries

load_libraries()

number_of_qubits = 2
resonators_original_frequencies = np.array([6, 7])  # GHz
frequencies_shifts = np.array([200, 200]) * 1e-6  # GHz

version_name = "_".join([f"({x}±{round(y / 2, 9)})_GHz" for x, y in zip(resonators_original_frequencies, frequencies_shifts)]) + "-v0"

assert number_of_qubits == len(resonators_original_frequencies) == len(
    frequencies_shifts), "qubits_frequencies or qubits_frequencies has wrong sizes"

configurations_frequencies = []

configurations_names = list(product(['0', '1'], repeat=number_of_qubits))
configurations_names = [''.join(item) for item in configurations_names]

temp_multiplication_factors = list(product([1, -1], repeat=number_of_qubits))
for multiplication_factors in temp_multiplication_factors:
    state_frequencies = []
    for qubits_frequency, frequencies_shift, multiplication_factor in zip(resonators_original_frequencies,
                                                                          frequencies_shifts, multiplication_factors):
        state_frequencies.append(qubits_frequency + multiplication_factor * frequencies_shift / 2)
    configurations_frequencies.append(state_frequencies)


def create_configuration(folder_name: str, configuration_name: str, resonator_frequencies: np.ndarray):
    max_meanders = 10
    min_segment_len = 150
    coupling_ground = 78
    airbridges = False

    chip_x = 6000
    chip_y = 10000
    tl_length = 2 * 4300

    # define save path
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "generated_chips", folder_name)
    os.makedirs(path, exist_ok=True)

    # create new layout
    lay = pya.Layout()
    top = lay.create_cell("TOP")
    pya.CellView.active().cell = lay.cell("TOP")

    # add chip to cell
    chip = lay.create_cell(
        "Chip",
        "WMIChips",
        {"chip_type": 1, "offset": 200}, # type 1 = "2c6x10"
    )
    trans = pya.DCplxTrans(1, 0, False, -chip_x / 2, -chip_y / 2)
    top.insert(pya.DCellInstArray(chip.cell_index(), trans))

    # add transmission line to cell
    transmission_line = lay.create_cell(
        "Cpw",
        "WMICPW",
        {
            "airbridge": False,
            "airbridge_spacing": 200,
            "path": pya.DPath([pya.DPoint(0, -tl_length / 2), pya.DPoint(0, tl_length / 2)], 0),
        },
    )
    trans = pya.DCplxTrans(1, 0, False, 0, 0)
    top.insert(pya.DCellInstArray(transmission_line.cell_index(), trans))

    # pre-generate all needed resonators
    resonator_cell_list = []
    total_cell_height = 0
    for frequency in resonator_frequencies:
        cell = lay.create_cell(
            "FabricationResonator",
            "WMITestStructures",
            {
                "frequency": frequency,
                "q_ext": 15,
                "start_termination": True,
                "coupling_ground": coupling_ground,
                "vert_offset": 200,
                "airbridge_spacing": 200,
                "airbridge_inset": 200,
                "airbridge": airbridges,
                "meanders": max_meanders,
                "min_segment_len": min_segment_len,
            },
        )
        resonator_cell_list.append(cell)
        total_cell_height += cell.bbox(lay.layer(default_layers.base_metal_gap.layer_info())).height()

    # calculate spacing of resonators
    spacing = (tl_length - total_cell_height * lay.dbu / 2) / (len(resonator_frequencies) + 1)
    y_pos = -tl_length / 2 + spacing

    # place resonator cells along the transmission line
    mirror = False
    for resonator in resonator_cell_list:
        trans = pya.DCplxTrans(1, 0 if not mirror else 180, mirror, 0, y_pos)
        top.insert(pya.DCellInstArray(resonator.cell_index(), trans))
        y_pos += (
                resonator.bbox(lay.layer(default_layers.base_metal_gap.layer_info())).height()
                / 2
                * lay.dbu
                + spacing
        )
        mirror = not mirror

    # place descriptive text on the chip
    text = lay.create_cell("Text", "WMIMisc", {"text": 'states=' + configuration_name})
    trans = pya.DCplxTrans(
        1,
        90,
        False,
        chip_x / 2 - 350,
        -text.bbox(lay.layer(default_layers.base_metal_gap.layer_info())).width() * lay.dbu / 2,
    )
    top.insert(pya.DCellInstArray(text.cell_index(), trans))

    # run export functions
    # export_layout(lay, configuration_name, path, format="json")
    export_layout(lay, configuration_name, path, format="gds")
    export_layout(lay, configuration_name, path, format="svg")
    # export_layout(lay, configuration_name, path, format="fab")


for frequencies, name in tqdm(zip(configurations_frequencies, configurations_names)):
    create_configuration(folder_name=version_name, configuration_name=name, resonator_frequencies=frequencies)
