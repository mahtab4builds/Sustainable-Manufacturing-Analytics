"""
Reusable plotting utilities for the project.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def save_chart(
    figure,
    output_path,
    dpi=150
):
    """
    Save a matplotlib figure using a standardized format.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close(figure)


def apply_standard_axis_labels(
    axis,
    title=None,
    xlabel=None,
    ylabel=None
):
    """
    Apply consistent chart labeling.
    """

    if title:
        axis.set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

    if xlabel:
        axis.set_xlabel(xlabel)

    if ylabel:
        axis.set_ylabel(ylabel)

    axis.grid(
        axis="y",
        alpha=0.25
    )


def create_standard_figure(
    width=10,
    height=6
):
    """
    Create a standardized matplotlib figure.
    """

    return plt.subplots(
        figsize=(width, height)
    )