# Machine Learning Midterm — Breast Cancer Wisconsin & Wine Classification

Decision Tree, KNN, Gaussian Naive Bayes and K-Means experiments on two scikit-learn datasets
(3 seeds, 132 classifier fits + 12 K-Means fits = 144 model fits).

## Project structure
| Path | Content |
|---|---|
| `breast_cancer.ipynb` | Sections 1–10 for Breast Cancer Wisconsin |
| `wine.ipynb` | Sections 1–10 for Wine Classification |
| `helpers.py` | Shared functions used by both notebooks |
| `results/` | Six combined CSV files (+ split indices and per-dataset parts) |
| `figures/` | All figures (PNG) |
| `report.docx` | Report |
| `AI_DISCLOSURE.md` | AI/tool-use disclosure |

## Installation and running (Windows / PowerShell)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```
Open each notebook and run **Kernel → Restart & Run All**. Run `breast_cancer.ipynb` and `wine.ipynb`
(in any order); after both have run, the six CSV files in `results/` contain both datasets.

## Environment
- Python: 3.13.x  <!-- TODO: write your exact version (python --version) -->
- Exact package versions: see `requirements.txt` (also printed in the first cell of each notebook).

## Data sources
- Breast Cancer Wisconsin (Diagnostic): `sklearn.datasets.load_breast_cancer`
- Wine: `sklearn.datasets.load_wine`
