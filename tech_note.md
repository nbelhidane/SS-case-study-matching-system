# Technical Note — Candidate–Job Matching System

## Problem framing

Given a candidate profile and a job posting, predict a match quality score (0–1) and rank candidates for a job or jobs for a candidate. The dataset is 9,460 candidate-job pairs across 28 job titles and 341 unique candidates — a near-complete 341×28 matrix, meaning almost every candidate was scored against every job. This structure makes the dataset unusual: it is not a sparse real-world matching table but a deliberately constructed benchmark.

## Data issues found

The `matched_score` label is quasi-discrete. There are 345 unique values across 9,460 rows; the two most common (0.85 and 0.65) account for 29.3% of all pairs. The label was computed by a rule-based algorithm, not human assessors, which sets a ceiling on regression performance independent of model quality.

Candidate skills and job skills share almost no vocabulary. After normalisation, only 5.8% of pairs have any token overlap (Jaccard > 0). Job `skills_required` uses composite phrases as single tokens — `'ccna (cisco certified network associate)'`, `'asp.net mvc strong understanding of database design'` — that never match candidate skill tokens regardless of cleaning. The vocabulary mismatch is structural. 21.4% of jobs have no `skills_required` listed at all; these jobs receive inflated scores because the algorithm cannot penalise skill mismatches.

Both `responsibilities` columns are identical (100% confirmed, 9,460/9,460 rows). Both carry the job description, not the candidate's work history. There is no candidate-side responsibilities field in this dataset.

Several candidate fields had structural nulls. `career_objective` is 50% null (empty string). Certification fields are 79% null with no usable content. Languages and address are 93% and 92% null respectively with no job-side counterparts. Educational results use incomparable grading systems (GPA/4, CGPA/10, percentage) — dropped on data quality and bias grounds. Age requirement was dropped as a protected characteristic.

## Feature engineering decisions

`candidate_doc` concatenates career objective, skills (lowercased, deduplicated), positions, and `related_skils_in_job`. Job skills in `candidate_doc` and `job_doc` were additionally cleaned — bullet characters stripped, compound tokens split on `or`/`and`/`/`, noise phrases removed. Skills in `job_doc` are the cleaned `skills_required` tokens.

Structured features: `exp_deficit` (years short of requirement), `exp_surplus` (years over), `years_experience` (inferred from date parsing, capped at 25), `edu_match` (degree ordinal ≥ requirement), `skill_coverage` (token overlap / job skills), `fuzzy_skill_coverage` (rapidfuzz token_sort_ratio ≥ 90), `skills_required_count` (proxy for the missing-requirement artefact), and `title_semantic_sim` (JobBERT-v3 cosine between candidate positions and job title — Spearman 0.25 with score, the single strongest individual feature), and `is_fresher` (binary flag from career_objective text — freshers score 0.56 vs 0.67 for non-freshers, Spearman −0.16).

## Model results

| | Model 0 (heuristic) | Model A (TF-IDF + Ridge) | Model B (MiniLM + Ridge) | Cross-encoder + Ridge |
|---|---|---|---|---|
| MAE | 0.474 | 0.127 | 0.125 | 0.131 |
| RMSE | 0.514 | 0.149 | 0.148 | 0.151 |
| Spearman r | 0.165 | 0.462 | 0.454 | **0.482** |
| NDCG@5 (by job) | 0.856 | 0.875 | 0.886 | 0.848 |

Model 0 fails because skill_coverage fires for only 6% of pairs, reducing the heuristic to an experience + title signal for most rows. Model B wins MAE and NDCG@5 — dense embeddings bridge the vocabulary gap that TF-IDF cannot. The cross-encoder wins Spearman (0.482) — cross-attention over the full pair provides the richest ranking signal at the cost of O(n×m) inference. Model A Spearman (0.462) is competitive for a sparse model; ablation without job_id raises it to 0.510, confirming TF-IDF has genuine cross-job signal. Model B drops to 0.433 without job_id, confirming HGB uses the job_id dummies as primary regression anchors.

## Limitations

The label is rule-based and quasi-discrete. The irreducible RMSE floor is approximately half the minimum gap between score levels. The theoretical Spearman ceiling given the available features is approximately 0.55–0.60; we reach 0.482 with the cross-encoder.

The train/test split is group-aware by job (80/20), but all 341 candidate profiles appear in both train and test across different jobs. Candidate feature leakage is inherent to the dataset structure. Reported metrics overestimate generalisation to truly unseen candidates.

NDCG@5 is dominated by jobs with wide score spreads. The Database Administrator job (std = 0.063) contributes near-random NDCG regardless of model quality.

## Production architecture (summary)

A two-stage retrieval and re-ranking system. Offline, encode all candidate profiles and job postings with JobBERT-v3 and store embeddings in a vector database (Qdrant or pgvector). At query time, stage one runs approximate nearest-neighbour search to retrieve the top-50 candidates in milliseconds — the recall stage. Stage two runs the full Ridge/HGB model with all structured features (exp_deficit, title_semantic_sim, skill_coverage, etc.) on the retrieved set — the precision stage. Full latency target under 200ms end-to-end. See `production_design.md` for the complete architecture.

## Given more time

An ESCO taxonomy layer mapping skills, education, and job titles to standardised identifiers would resolve the vocabulary mismatch structurally — the right long-term solution. Fine-tuning a sentence transformer with contrastive loss on (candidate, job, score) triples would make embeddings respond to the specific matching criteria in the label. NER on `educational_requirements` would extract degree level and field precisely. Per-field similarity (skills vs skills, title vs title) would give the model more focused signals than full-document cosine. A production ANN index with re-ranking would enable sub-second retrieval at scale.
