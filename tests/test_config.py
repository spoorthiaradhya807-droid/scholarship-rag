"""Settings must be sensible and defined once."""
import re
from pathlib import Path

from src import config


def test_thresholds_are_sane():
    assert 0 < config.RELEVANCE_MIN < 1
    assert 0 < config.LIST_RELEVANCE_MIN <= config.RELEVANCE_MIN
    assert config.LIST_TOP_K > config.TOP_K > 0


def test_each_setting_is_defined_only_once():
    source = Path(config.__file__).read_text(encoding="utf-8")
    names = re.findall(r"^([A-Z_]+)\s*=", source, re.M)
    assert len(names) == len(set(names))