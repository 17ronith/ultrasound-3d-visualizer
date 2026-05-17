"""Playwright tests for the Pulse Analysis feature."""
import pytest
from playwright.sync_api import Page, expect

BASE = 'http://localhost:8000'
MHD  = 'FAST/data/US/JugularVein/US-2D_0.mhd'
RAW  = 'FAST/data/US/JugularVein/US-2D_0.raw'


@pytest.fixture(scope='module', autouse=True)
def ensure_server():
    """Server must be running before these tests. Start with: bash start.sh"""
    import requests
    try:
        requests.get(BASE, timeout=3)
    except Exception:
        pytest.skip('Server not running — start with: bash start.sh')


def upload_jugular(page: Page):
    """Upload JugularVein MHD + RAW files with anatomy selected, reach mode_select."""
    page.goto(BASE)
    page.select_option('#anatomy-select', 'Jugular Vein')
    with page.expect_file_chooser() as fc_info:
        page.click('#drop-zone')
    fc = fc_info.value
    fc.set_files([MHD, RAW])
    # Wait for mode_select to appear (inference takes a few seconds)
    page.wait_for_selector('#state-mode-select', state='visible', timeout=30_000)


def test_mode_select_appears_after_upload(page: Page):
    upload_jugular(page)
    expect(page.locator('#state-mode-select')).to_be_visible()
    expect(page.locator('#mode-btn-single')).to_be_visible()
    expect(page.locator('#mode-btn-pulse')).to_be_visible()


def test_pulse_card_enabled_for_jugular(page: Page):
    upload_jugular(page)
    pulse_btn = page.locator('#mode-btn-pulse')
    expect(pulse_btn).not_to_be_disabled()
    pulse_card = page.locator('#mode-card-pulse')
    expect(pulse_card).not_to_have_class('mode-card--disabled')


def test_single_frame_navigates_to_dashboard(page: Page):
    upload_jugular(page)
    page.click('#mode-btn-single')
    expect(page.locator('#state-dashboard')).to_be_visible()
    expect(page.locator('#three-container canvas').first).to_be_visible()


def test_back_link_from_mode_select_resets(page: Page):
    upload_jugular(page)
    page.click('#mode-back-link')
    expect(page.locator('#state-drop')).to_be_visible()


def test_pulse_analysis_renders_pulse_state(page: Page):
    upload_jugular(page)
    page.click('#mode-btn-pulse')
    # Pulse analysis takes ~60s — use generous timeout
    page.wait_for_selector('#state-pulse', state='visible', timeout=120_000)
    expect(page.locator('#pulse-header-badges')).to_contain_text('JugularVein')
    expect(page.locator('#pulse-stats-row')).to_contain_text('BPM')


def test_pulse_scrubber_updates_frame_label(page: Page):
    upload_jugular(page)
    page.click('#mode-btn-pulse')
    page.wait_for_selector('#state-pulse', state='visible', timeout=120_000)
    page.evaluate("document.getElementById('pulse-scrubber').value = 50; document.getElementById('pulse-scrubber').dispatchEvent(new Event('input'))")
    expect(page.locator('#pulse-frame-label')).to_contain_text('50')


def test_pulse_back_returns_to_mode_select(page: Page):
    upload_jugular(page)
    page.click('#mode-btn-pulse')
    page.wait_for_selector('#state-pulse', state='visible', timeout=120_000)
    page.click('#pulse-back-link')
    expect(page.locator('#state-mode-select')).to_be_visible()
    expect(page.locator('#mode-btn-single')).to_be_visible()
