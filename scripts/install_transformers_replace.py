import shutil
from pathlib import Path
import transformers

def copy_tree(src: Path, dst: Path):
    for p in src.rglob("*.py"):
        rel = p.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)
        print("copied:", rel)

if __name__ == "__main__":
    assert transformers.__version__ == "4.53.2", f"transformers==4.53.2 required, got {transformers.__version__}"
    repo_root = Path(__file__).resolve().parents[1]
    src_root  = repo_root / "src" / "openpi" / "models_pytorch" / "transformers_replace" / "models"
    site_root = Path(transformers.__file__).resolve().parent / "models"
    print("FROM:", src_root)
    print("TO  :", site_root)
    copy_tree(src_root, site_root)
    print("done.")