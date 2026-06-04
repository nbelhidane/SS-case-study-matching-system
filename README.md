# Candidate–Job Matching System

Spencer Stuart case study. Predicts match quality between a candidate profile and a job posting, and ranks candidates for a job or jobs for a candidate.

## Setup

```bash
# Create environment (Python 3.13 via Homebrew)
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register as a Jupyter kernel:
```bash
python -m ipykernel install --user --name case-study-ss --display-name "Case Study (Python 3.13)"
```

Then select **Case Study (Python 3.13)** in VS Code's kernel picker.

## Run order

Run notebooks in this exact order. Each produces outputs consumed by the next.

| Step | Notebook | Input | Output |
|---|---|---|---|
| 1 | `01_eda.ipynb` | `data/resume.csv` | findings documented in FINDINGS.md |
| 2 | `02_features.ipynb` | `data/resume.csv` | `outputs/features.csv`, `outputs/title_vecs_*.npy` |
| 3 | `03_model0.ipynb` | `outputs/features.csv` | `outputs/model0_predictions.csv` |
| 4 | `03_model_a.ipynb` | `outputs/features.csv` | `outputs/model_a_predictions.csv` |
| 5 | `03_model_b.ipynb` | `outputs/features.csv`, cached vectors | `outputs/model_b_predictions.csv` |
| 6 | `04_evaluation.ipynb` | all prediction CSVs | `outputs/viz_*.png` |
| 7 | `05_examples.ipynb` | prediction CSVs, features.csv | — |
| 8 | `06_visualizations.ipynb` | features.csv, cached vectors, prediction CSVs | `outputs/viz_umap_embeddings.png`, `viz_vocab_venn.png`, `viz_feature_importance.png`, `viz_score_dist.png`, `viz_pred_vs_true.png` |

**Note:** `03_model_b.ipynb` encodes ~9,500 documents with two sentence transformer models. First run takes 10–20 minutes on CPU. Vectors are cached to `outputs/*.npy` and reloaded on subsequent runs.

## Repository structure

```
├── data/
│   └── resume.csv               # raw dataset (9,544 rows, 35 columns)
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory analysis
│   ├── 02_features.ipynb        # feature engineering pipeline
│   ├── 03_model0.ipynb          # heuristic baseline
│   ├── 03_model_a.ipynb         # TF-IDF + Ridge
│   ├── 03_model_b.ipynb         # JobBERT + HistGradientBoosting
│   ├── 04_evaluation.ipynb      # consolidated metrics and visualisations
│   ├── 05_examples.ipynb        # concrete match/mismatch examples
│   └── 06_visualizations.ipynb  # publication-ready figures (UMAP, Venn, importances)
├── outputs/
│   ├── features.csv             # 9,460 rows × 27 features
│   ├── model0_predictions.csv
│   ├── model_a_predictions.csv
│   ├── model_b_predictions.csv
│   ├── ablation_a.json          # job_id ablation results (Model A)
│   ├── ablation_b.json          # job_id ablation results (Model B)
│   └── viz_*.png                # evaluation plots
├── PLAN.md                      # project plan and column decisions
├── FINDINGS.md                  # EDA findings with decisions
├── SUMMARY.md                   # full narrative of everything built
├── tech_note.md                 # one-page technical summary
├── production_design.md         # production architecture
└── requirements.txt
```

## Requirements

See `requirements.txt`. Key packages:
- `sentence-transformers>=5.5` (JobBERT-v3 requires this version)
- `rapidfuzz` (fuzzy skill matching)
- `rank-bm25` (BM25 experiment in Model A)
- `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`
- `torch`, `transformers`

## Key results

| Model | MAE | Spearman r | NDCG@5 |
|---|---|---|---|
| Model 0 (heuristic) | 0.474 | 0.165 | 0.856 |
| Model A (TF-IDF + Ridge) | 0.127 | 0.462 | 0.875 |
| Model B (MiniLM + Ridge) | 0.125 | 0.454 | 0.886 |
| Cross-encoder + Ridge | 0.131 | **0.482** | 0.848 |

Model B wins on MAE and NDCG@5 (within-job ranking). Model A wins on Spearman (cross-job). See `tech_note.md` for full analysis and `production_design.md` for deployment architecture.
