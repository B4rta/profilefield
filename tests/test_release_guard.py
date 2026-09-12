from pathlib import Path
from runpy import run_path


def test_release_guard_rejects_private_manuscript_and_data_paths() -> None:
    guard = run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_release.py"))
    public = ["README.md", "data/README.md", "src/profilefield/artifacts.py", "third_party/geovae-sgs"]
    private = ["manuscript/draft.md", "docs/summary.md", "paper_results/table.csv", "plots/panel.svg", "data/secret.parquet", "subdir/.env.local"]
    assert guard["forbidden_paths"](public + private) == sorted(private)
