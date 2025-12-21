# analysis/plotting/groupcomparison/api.py

import matplotlib.pyplot as plt
from .layouts import plot_views_as_rows_1x3, plot_abls_as_rows_4x3

def plot_groupcomparison(
    bundle,
    *,
    layout="views_1x3",
    makefig1_data=None,
    makefig1_chrono=None,
    old_jnd_data=None,
    show=True,
    **kwargs,
):
    """
    Returns:
      - if layout == "views_1x3": (main_fig, jnd_figs)
      - else: main_fig
    """
    if layout == "views_1x3":
        out = plot_views_as_rows_1x3(
            bundle,
            makefig1_data=makefig1_data,
            makefig1_chrono=makefig1_chrono,
            old_jnd_data=old_jnd_data,
            **kwargs,
        )
        # out is (main_fig, jnd_figs)
        main_fig, jnd_figs = out

        if show:
            plt.show()  # shows main + jnd figs, once, at the end

        return main_fig, jnd_figs

    elif layout == "abls_4x3":
        main_fig = plot_abls_as_rows_4x3(
            bundle,
            makefig1_data=makefig1_data,
            makefig1_chrono=makefig1_chrono,
            **kwargs,                       # <<< CRITICAL
        )
        if show:

            plt.show()
        return main_fig
    
    elif layout == "multibundle_abls_4x3":
        from .layouts_multi import plot_multibundle_abls_4x3
        fig = plot_multibundle_abls_4x3(**kwargs)
        if show:

            plt.show()
        return fig



    else:
        raise ValueError(f"Unknown layout: {layout}")
