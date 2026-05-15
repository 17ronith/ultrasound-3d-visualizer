"""
Ex 28 — GUI with a single slider widget to control NLM smoothing amount.
Loads a single static frame from Heart sequence.
"""
from _config import fast, DATA

importer = fast.ImageFileImporter.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_0.mhd'
)

nlm = fast.NonLocalMeans.create(
    filterSize=3,
    searchSize=11,
    smoothingAmount=0.2,
    inputMultiplicationWeight=0.5,
).connect(importer)

sliderWidget = fast.SliderWidget(
    'Smoothing', 0.2, 0.05, 0.8, 0.05,
    fast.SliderCallback(lambda x: nlm.setSmoothingAmount(x)),
)
fast.display2D(nlm, widgets=[sliderWidget])
