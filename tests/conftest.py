import pytest
import numpy as np
import fast

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')
DATA = fast.Config.getTestDataPath()


@pytest.fixture
def data_path():
    return DATA


@pytest.fixture
def jugular_mhd_path():
    return DATA + 'US/JugularVein/US-2D_0.mhd'


@pytest.fixture
def single_jpg_path():
    return DATA + 'US/US-2D.jpg'


@pytest.fixture
def ball_mhd_path():
    return DATA + 'US/Ball/US-3Dt_0.mhd'


@pytest.fixture
def gray_array_uint8():
    return np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)


@pytest.fixture
def gray_array_float():
    return np.random.rand(64, 64, 1).astype(np.float32)


@pytest.fixture
def binary_mask():
    mask = np.zeros((64, 64, 1), dtype=np.uint8)
    mask[20:44, 20:44] = 1
    return mask
