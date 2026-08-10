"""Publication-ready figure export helpers.

The export boundary deliberately accepts Matplotlib figures instead of rebuilding
plots.  This keeps the on-screen preview and downloaded artwork identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from matplotlib.figure import Figure


@dataclass(frozen=True)
class FigureFile:
    """A named, in-memory figure ready for a Streamlit download button."""

    name: str
    png: bytes
    tiff: bytes
    svg: bytes
    pdf: bytes


def safe_filename(value: object) -> str:
    """Return a portable, readable filename stem."""
    text = str(value).strip().replace("œ", "oe").replace("Œ", "OE")
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")
    return text or "figure"


def figure_bytes(figure: Figure, format: str, *, dpi: int = 600) -> bytes:
    """Serialize a figure with the tight, white layout used by example scripts."""
    output = BytesIO()
    metadata = {"Creator": "LuxPlate Analyzer"} if format in {"png", "pdf", "svg"} else None
    figure.savefig(
        output, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0.03,
        facecolor="white", metadata=metadata,
    )
    return output.getvalue()


def package_figures(figures: list[tuple[str, Figure]], *, dpi: int = 600) -> tuple[list[FigureFile], bytes]:
    """Render figures to raster and vector formats and return a ZIP archive."""
    rendered: list[FigureFile] = []
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, figure in figures:
            stem = safe_filename(Path(name).stem)
            item = FigureFile(stem, figure_bytes(figure, "png", dpi=dpi),
                              figure_bytes(figure, "tiff", dpi=dpi),
                              figure_bytes(figure, "svg", dpi=dpi),
                              figure_bytes(figure, "pdf", dpi=dpi))
            rendered.append(item)
            bundle.writestr(f"PNG/{stem}.png", item.png)
            bundle.writestr(f"TIFF/{stem}.tiff", item.tiff)
            bundle.writestr(f"SVG/{stem}.svg", item.svg)
            bundle.writestr(f"PDF/{stem}.pdf", item.pdf)
        bundle.writestr(
            "LISEZ_MOI.txt",
            f"Figures LuxPlate prêtes pour publication.\nPNG et TIFF : images {dpi} dpi.\n"
            "SVG et PDF : formats vectoriels éditables.\n",
        )
    return rendered, archive.getvalue()
