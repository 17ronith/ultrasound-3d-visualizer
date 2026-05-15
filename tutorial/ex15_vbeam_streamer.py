"""
Ex 15 & 16 — VBeam streamer + DAS beamforming pipeline.

SKIPPED: requires extra packages not in this venv:
  pip install vbeam jax pyuff-ustb

Once installed, download the VbeamStreamer class from the tutorial page and run:
  python ex15_vbeam_streamer.py

The pipeline beamforms raw channel data from an external UFF dataset using
the VBeam DAS beamformer, applies envelope detection, scan conversion, and
NonLocalMeans denoising with interactive sliders.
"""
print(__doc__)
