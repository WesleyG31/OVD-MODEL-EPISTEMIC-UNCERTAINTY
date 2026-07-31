from pathlib import Path

from scripts.run_rq1 import default_config_path


def test_rq1_mode_selects_isolated_configuration() -> None:
    root = Path("project")
    assert default_config_path(root, "mini").name == "rq1_mini.yaml"
    assert default_config_path(root, "smoke").name == "rq1.yaml"
    assert default_config_path(root, "full").name == "rq1.yaml"
