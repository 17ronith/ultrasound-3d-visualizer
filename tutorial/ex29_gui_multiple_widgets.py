"""
Ex 29 — GUI with multiple widgets: smoothing, filter size, search size,
input multiplication weight sliders, and a toggle button.
Side-by-side original vs NLM-denoised with interactive controls.
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

renderer = fast.ImageRenderer.create().connect(importer)
rendererNLM = fast.ImageRenderer.create().connect(nlm)

sliderWidget = fast.SliderWidget('Smoothing', 0.2, 0.05, 0.8, 0.05,
    fast.SliderCallback(lambda x: nlm.setSmoothingAmount(x)))
filterWidget = fast.SliderWidget('Filter size', 3, 3, 19, 2,
    fast.SliderCallback(lambda x: nlm.setFilterSize(int(x))))
searchWidget = fast.SliderWidget('Search size', 11, 3, 19, 2,
    fast.SliderCallback(lambda x: nlm.setSearchSize(int(x))))
inputWidget = fast.SliderWidget('Input mult. weight', 0.5, 0.0, 1.0, 0.05,
    fast.SliderCallback(lambda x: nlm.setInputMultiplicationWeight(x)))
toggleButton = fast.ButtonWidget('Toggle ON/OFF', True,
    fast.ButtonCallback(lambda x: rendererNLM.setDisabled(x)))

fast.SimpleWindow2D.create()\
    .connect(renderer)\
    .connect(rendererNLM)\
    .connect([sliderWidget, filterWidget, searchWidget, inputWidget, toggleButton],
             fast.WidgetPosition_RIGHT)\
    .run()
