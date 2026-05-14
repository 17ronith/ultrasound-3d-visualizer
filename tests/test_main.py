# tests/test_main.py
import subprocess
import sys
import os


def test_main_runs_without_error_on_jugular_mhd(data_path):
    env = os.environ.copy()
    env['ULTRASOUND_FILEPATH'] = data_path + 'US/JugularVein/US-2D_0.mhd'
    env['ULTRASOUND_NO_BROWSER'] = '1'  # skip browser open in CI
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert 'JugularVein' in result.stdout
    assert 'jugular_vein_segmentation.onnx' in result.stdout


def test_main_runs_on_ball_3d(data_path):
    env = os.environ.copy()
    env['ULTRASOUND_FILEPATH'] = data_path + 'US/Ball/US-3Dt_0.mhd'
    env['ULTRASOUND_NO_BROWSER'] = '1'
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert 'Ball' in result.stdout
    assert '3D_volume' in result.stdout


def test_main_runs_on_plain_jpg(data_path):
    env = os.environ.copy()
    env['ULTRASOUND_FILEPATH'] = data_path + 'US/US-2D.jpg'
    env['ULTRASOUND_NO_BROWSER'] = '1'
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert 'unknown' in result.stdout
    assert 'fallback' in result.stdout.lower() or 'intensity' in result.stdout.lower()
