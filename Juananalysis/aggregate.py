# aggregate.py
import numpy as np

def aggregate_RT_MT_Out(df, animals=None):
    """Organizes RT, MT, and Correct into the structure expected by kernel_regression."""

    if animals is None:
        animals = df["animal"].unique()

    ABLs = df["ABL"].dropna().unique()
    ABLs.sort()

    result = {
        "RTs": {},
        "MTs": {},
        "Outs": {},
        "animals": animals,
        "ABLs": ABLs,
    }

    # Loop: LED (0/1), Animal, ABL, Sign
    for led_val in [0, 1]:
        result["RTs"][led_val] = []
        result["MTs"][led_val] = []
        result["Outs"][led_val] = []

        for animal in animals:
            df_an = df[(df["animal"] == animal) & (df["LED"] == led_val)]
            RT_list = []
            MT_list = []
            Out_list = []

            for abl in ABLs:
                df_ab = df_an[df_an["ABL"] == abl]

                # Sign: +1 (right), -1 (left)
                cond_R = df_ab[df_ab["Sign"] == 1]
                cond_L = df_ab[df_ab["Sign"] == -1]

                RT_list.append([cond_R["RT"].dropna().values,
                                cond_L["RT"].dropna().values])
                MT_list.append([cond_R["MT"].dropna().values,
                                cond_L["MT"].dropna().values])
                Out_list.append([cond_R["Out"].dropna().values,
                                 cond_L["Out"].dropna().values])

            result["RTs"][led_val].append(RT_list)
            result["MTs"][led_val].append(MT_list)
            result["Outs"][led_val].append(Out_list)

    return result
