"""Regression checks for the installation drift that broke native VLM startup."""

from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _requirements(path):
    return [
        Requirement(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_direct_server_extras_match_complete_runtime_lock():
    lock = {
        requirement.name.lower().replace("_", "-"): requirement
        for requirement in _requirements(ROOT / "requirements/server-lock.txt")
    }
    for file in ("server.in", "vision.in", "classifier.in"):
        for direct in _requirements(ROOT / "requirements" / file):
            locked = lock[direct.name.lower().replace("_", "-")]
            versions = [specifier.version for specifier in locked.specifier]
            assert len(versions) == 1
            assert direct.specifier.contains(versions[0], prereleases=True), (direct, locked)


def test_docker_consumes_shared_lock_and_retains_flash_attention():
    dockerfile = (ROOT / "docker/Dockerfile-cu124").read_text()
    assert "pip install -r requirements-server.txt" in dockerfile
    assert "pip install --no-build-isolation -r requirements-flash-attn.txt" in dockerfile
    assert "flash-attn==2.7.3" in (ROOT / "requirements-flash-attn.txt").read_text()


def test_runtime_lock_has_only_exact_pins():
    for requirement in _requirements(ROOT / "requirements/server-lock.txt"):
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1
        assert specifiers[0].operator == "=="
        assert "*" not in specifiers[0].version


def test_windows_lock_omits_linux_extensions_but_retains_cuda_torch_build():
    environment = default_environment()
    environment.update(sys_platform="win32", platform_system="Windows", platform_machine="AMD64")
    selected = {
        requirement.name: requirement
        for requirement in _requirements(ROOT / "requirements/server-lock.txt")
        if not requirement.marker or requirement.marker.evaluate(environment)
    }
    assert "uvloop" not in selected
    assert "triton" not in selected
    assert not any(name.startswith("nvidia-") for name in selected)
    assert str(selected["torch"].specifier) == "==2.4.1+cu124"
    assert str(selected["torchvision"].specifier) == "==0.19.1+cu124"


def test_eager_import_filter_is_scoped_and_restored(tmp_path):
    import pytest
    from unittest.mock import patch

    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    from transformers import dynamic_module_utils

    from enose.vision.florence import _eager_imports_without_flash_attention

    content = "from torch import nn\nif is_flash_attn_2_available():\n    from flash_attn import helper\n"
    florence = tmp_path / "modeling_florence2.py"
    other = tmp_path / "other_model.py"
    florence.write_text(content)
    other.write_text(content)
    # Newer transformers scanners already omit conditional imports. Model the
    # pinned 4.44.2 result so this regression still tests the problematic case.
    with patch.object(dynamic_module_utils, "get_imports", return_value=["torch", "flash_attn"]):
        original = dynamic_module_utils.get_imports
        with _eager_imports_without_flash_attention():
            assert "flash_attn" not in dynamic_module_utils.get_imports(florence)
            assert "torch" in dynamic_module_utils.get_imports(florence)
            assert "flash_attn" in dynamic_module_utils.get_imports(other)
        assert dynamic_module_utils.get_imports is original
