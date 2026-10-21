"""
tests/test_apptest_smoke.py - Smoke test each level with streamlit.testing.v1.AppTest.
Verifies that the three baseline cards equal the README seed-42 baseline numbers:
- Level 1: Val 88.3%, Test Avg 51.9%, Worst-Group 7.8%
- Level 2: Val 76.8%, Test Avg 60.3%, Worst-Group 39.5%
- Level 3: Val 78.0%, Test Avg 59.4%, Worst-Group 32.6%
"""
from streamlit.testing.v1 import AppTest
import re
import pytest

EXPECTED_BASELINE = {
    1: {"val": "88.3%", "test": "51.9%", "wg": "7.8%"},
    2: {"val": "76.8%", "test": "60.3%", "wg": "39.5%"},
    3: {"val": "78.0%", "test": "59.4%", "wg": "32.6%"},
}


def _verify_level(level: int):
    at = AppTest.from_file("streamlit_app.py", default_timeout=300)
    # Set initial active level in session state before first run
    at.session_state["active_level"] = level
    at.run()
    assert not at.exception, f"App raised exception on level {level}: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    full_text = "\n".join(md_texts)

    exp = EXPECTED_BASELINE[level]
    val_pattern = rf"<div class='mbig'>{re.escape(exp['val'])}</div>\s*<div class='mlbl'>Baseline Val Acc</div>"
    test_pattern = rf"<div class='mbig'>{re.escape(exp['test'])}</div>\s*<div class='mlbl'>Baseline Test Avg</div>"
    wg_pattern = rf"<div class='mbig' style='color:#f87171'>\s*{re.escape(exp['wg'])}\s*</div>\s*<div class='mlbl'>Baseline Worst-Group</div>"

    assert re.search(val_pattern, full_text), f"Level {level} Baseline Val Acc card mismatch! Expected {exp['val']} in:\n{full_text}"
    assert re.search(test_pattern, full_text), f"Level {level} Baseline Test Avg card mismatch! Expected {exp['test']} in:\n{full_text}"
    assert re.search(wg_pattern, full_text), f"Level {level} Baseline Worst-Group card mismatch! Expected {exp['wg']} in:\n{full_text}"
    print(f"Level {level} verified via AppTest: Val={exp['val']}, Test={exp['test']}, WG={exp['wg']}")


def test_level1_baseline_cards():
    _verify_level(1)


def test_level2_baseline_cards():
    _verify_level(2)


def test_level3_baseline_cards():
    _verify_level(3)


if __name__ == "__main__":
    test_level1_baseline_cards()
    test_level2_baseline_cards()
    test_level3_baseline_cards()
    print("All AppTest level smoke tests passed successfully!")
