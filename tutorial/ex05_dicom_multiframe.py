"""
Ex 05 — Read DICOM multiframe file.
Uses the LIDC-IDRI-0072 CT lung dataset (first slice as a single-frame example).
Note: DICOMMultiFrameStreamer expects a multi-frame DICOM; LIDC has single-frame
slices per file, so this streams the first .dcm file.
"""
from _config import fast, DATA
import pathlib

dicom_dir = pathlib.Path(DATA) / "CT/LIDC-IDRI-0072"
first_dcm = sorted(dicom_dir.glob("*.dcm"))[0]
print(f"Loading DICOM: {first_dcm}")

streamer = fast.DICOMMultiFrameStreamer.create(
    str(first_dcm),
    loop=True,
    grayscale=False,
    cropToROI=True,
)

fast.display2D(streamer)
