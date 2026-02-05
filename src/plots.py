"""
Small plotting helpers.
"""

import matplotlib.pyplot as plt  # Plotting helpers.


def show_or_save(fig, savepath=None):
    """
    Unified figure display/save helper.

    If savepath is provided, save to disk; otherwise show interactively.
    """
    if savepath:  # Save to file.
        # dpi/bbox settings improve figure clarity and trim whitespace.
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    else:  # Show interactively.
        plt.show()
