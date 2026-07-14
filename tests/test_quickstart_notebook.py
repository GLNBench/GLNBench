import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "quickstart.ipynb"


def _code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_quickstart_accepts_current_colab_python():
    runtime_cell = _code_cells()[0]
    tree = ast.parse(runtime_cell)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "QUICKSTART_PYTHONS" for target in node.targets)
    )
    assert (3, 12) in ast.literal_eval(assignment.value)
    assert "pinned release supports Python 3.10/3.11" not in runtime_cell


def test_python_312_uses_colab_requirements():
    install_cell = _code_cells()[1]
    assert 'sys.version_info[:2] == (3, 12)' in install_cell
    assert 'requirements_file = "requirements-colab.txt"' in install_cell
    assert 'requirements_file = "requirements.txt"' in install_cell
    assert "subprocess.check_call" in install_cell


def test_colab_requirements_preserve_runtime_pytorch():
    requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip().lower() for line in requirements if line.strip() and not line.startswith("#")]
    assert "torch-geometric==2.7.0" in packages
    assert "cleanlab==2.7.1" in packages
    assert "scikit-learn>=1.4,<1.8" in packages
    assert not any(line.startswith(("torch==", "torch>=", "torch<=")) for line in packages)
