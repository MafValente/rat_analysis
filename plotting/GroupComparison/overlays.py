# analysis/plotting/groupcomparison/overlays.py

import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_makefig1_data(path):
    data = load_pickle(path)
    # --- Remap ABL 35 → 40 ---
    if 35 in data.get("ABLS", []):
        data["ABLS"] = [40 if x == 35 else x for x in data["ABLS"]]
        for key in ["ilds_dict", "mean_sigmoid_dict", "mean_params_dict", "x_smooth_dict"]:
            if key in data and 35 in data[key]:
                data[key][40] = data[key].pop(35)
    return data


def load_makefig1_chrono(path):
    data = load_pickle(path)
    if 35 in data.get("plot_abls", []):
        data["plot_abls"] = [40 if x == 35 else x for x in data["plot_abls"]]
        for key in ["ilds_dict", "mean_sigmoid_dict", "mean_params_dict", "x_smooth_dict"]:
            if key in data and 35 in data[key]:
                data[key][40] = data[key].pop(35)
    return data
