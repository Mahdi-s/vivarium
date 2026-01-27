"""
Centralized matplotlib style configuration for publication-ready figures.

Enforces PowerPoint + paper quality standards:
- PNG at 300+ DPI for slides
- PDF (vector) for paper
- Readable fonts, non-overlapping labels
- Consistent color palettes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    raise RuntimeError("matplotlib is required for plotting")


# Publication-quality font settings
PUBLICATION_RCPARAMS = {
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "patch.linewidth": 1.0,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
}

# Default figure sizes (inches) for different use cases
FIGURE_SIZES = {
    "single": (10, 6),  # Single panel, standard
    "dense": (12, 7),  # Dense legends, multiple series
    "compact": (7, 4),  # Compact inline
    "wide": (13.33, 7.5),  # 16:9 aspect ratio for slides
    "tall": (8, 10),  # Tall for multi-panel vertical
}

# Consistent color palette (colorblind-friendly)
COLOR_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def setup_publication_style() -> None:
    """Configure matplotlib rcParams for publication quality."""
    plt.rcParams.update(PUBLICATION_RCPARAMS)


def get_figure_size(size_key: str = "single") -> Tuple[float, float]:
    """Get figure size tuple for a given key."""
    return FIGURE_SIZES.get(size_key, FIGURE_SIZES["single"])


def save_figure(
    fig: Figure,
    base_path: str,
    formats: Optional[list[str]] = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
) -> Dict[str, str]:
    """
    Save figure in multiple formats (PNG + PDF) with publication quality.
    
    Args:
        fig: matplotlib Figure object
        base_path: Base path without extension (e.g., "/path/to/figure")
        formats: List of formats to save. Default: ["png", "pdf"]
        dpi: DPI for raster formats (PNG). Default: 300
        bbox_inches: bbox_inches parameter for savefig
        pad_inches: pad_inches parameter for savefig
        
    Returns:
        Dict mapping format -> saved path
    """
    if formats is None:
        formats = ["png", "pdf"]
    
    saved_paths = {}
    base = Path(base_path)
    
    # Ensure directory exists
    base.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        if fmt == "png":
            path = str(base.with_suffix(".png"))
            fig.savefig(
                path,
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="none",
            )
            saved_paths["png"] = path
        elif fmt == "pdf":
            path = str(base.with_suffix(".pdf"))
            fig.savefig(
                path,
                format="pdf",
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="none",
            )
            saved_paths["pdf"] = path
        elif fmt == "svg":
            path = str(base.with_suffix(".svg"))
            fig.savefig(
                path,
                format="svg",
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="none",
            )
            saved_paths["svg"] = path
    
    return saved_paths


def create_figure(
    size_key: str = "single",
    constrained_layout: bool = True,
    **kwargs: Any,
) -> Tuple[Figure, Any]:
    """
    Create a figure with publication-quality defaults.
    
    Args:
        size_key: Key for figure size (see FIGURE_SIZES)
        constrained_layout: Use constrained_layout for automatic spacing
        **kwargs: Additional arguments passed to plt.subplots
        
    Returns:
        (fig, ax) tuple
    """
    setup_publication_style()
    figsize = get_figure_size(size_key)
    
    if constrained_layout:
        kwargs.setdefault("constrained_layout", True)
    
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def rotate_labels_if_needed(ax: Any, axis: str = "x", max_length: int = 15, angle: int = 45) -> None:
    """
    Rotate axis labels if they're too long to prevent overlap.
    
    Args:
        ax: matplotlib axes
        axis: "x" or "y"
        max_length: Maximum label length before rotating
        angle: Rotation angle in degrees
    """
    if axis == "x":
        labels = [label.get_text() for label in ax.get_xticklabels()]
        if any(len(l) > max_length for l in labels):
            ax.tick_params(axis="x", rotation=angle)
            for label in ax.get_xticklabels():
                label.set_ha("right")
    elif axis == "y":
        labels = [label.get_text() for label in ax.get_yticklabels()]
        if any(len(l) > max_length for l in labels):
            ax.tick_params(axis="y", rotation=angle)


def wrap_long_labels(labels: list[str], max_length: int = 20) -> list[str]:
    """
    Wrap long labels by inserting newlines.
    
    Args:
        labels: List of label strings
        max_length: Maximum characters per line
        
    Returns:
        List of wrapped labels
    """
    wrapped = []
    for label in labels:
        if len(label) > max_length:
            # Simple word-wrap (split on spaces)
            words = label.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_length:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            wrapped.append("\n".join(lines))
        else:
            wrapped.append(label)
    return wrapped


def get_color_palette(n: int) -> list[str]:
    """
    Get a color palette with n colors, cycling if needed.
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of color strings
    """
    if n <= len(COLOR_PALETTE):
        return COLOR_PALETTE[:n]
    # Cycle if more colors needed
    return [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(n)]
