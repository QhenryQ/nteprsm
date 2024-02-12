import importlib.resources

SRC_DIR = importlib.resources.files("nteprsm")

# repository roo folder
ROOT_DIR = SRC_DIR.parent

# standard location for data files
DATA_DIR = ROOT_DIR / "data"

# standard location for model files (*.stan or *.pickle)
MODEL_DIR = ROOT_DIR / "models"

# standard location for report files, i.e., generated analysis as HTML, PDF, LaTeX, etc
REPORT_DIR = ROOT_DIR / "reports"
