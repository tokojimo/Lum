from io import BytesIO
from zipfile import ZipFile

import matplotlib.pyplot as plt

from luxplate.export import package_figures, safe_filename


def test_safe_filename_keeps_readable_stem():
    assert safe_filename("SCFM + glucose / croissance") == "SCFM_glucose_croissance"


def test_package_figures_contains_png_pdf_and_readme():
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    rendered, payload = package_figures([("Milieu 1", figure)], dpi=72)

    assert rendered[0].png.startswith(b"\x89PNG")
    assert rendered[0].pdf.startswith(b"%PDF")
    with ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "PNG/Milieu_1.png", "PDF/Milieu_1.pdf", "LISEZ_MOI.txt",
        }
    plt.close(figure)
