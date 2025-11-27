# plot_results.py
import matplotlib.pyplot as plt
import numpy as np

def shaded_curve(x, mean, up, dn, color="k", label=None):
    plt.plot(x, mean, color=color, label=label)
    plt.fill_between(x, mean - dn, mean + up,
                     color=color, alpha=0.25)

def plot_TCM(xxi, TCM_LED0, TCM_LED1):
    plt.figure(figsize=(6,4))
    shaded_curve(xxi, *TCM_LED0, color="k", label="LED OFF")
    shaded_curve(xxi, *TCM_LED1, color="blue", label="LED ON")
    plt.xlabel("RT (s)")
    plt.ylabel("P(correct)")
    plt.legend()
    plt.tight_layout()
    plt.show()
