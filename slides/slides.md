---
marp: true
theme: default
paginate: true
style: |
  section { font-family: 'Helvetica Neue', Arial, sans-serif; }
  h1 { color: #1a1a2e; font-size: 2em; }
  h2 { color: #16213e; font-size: 1.4em; }
  table { font-size: 0.85em; width: 100%; }
  th { background: #16213e; color: white; }
  .highlight { color: #e94560; font-weight: bold; }
  .dim { color: #888; font-size: 0.85em; }
---

# Candidate–Job Matching System

**Spencer Stuart Case Study**

Nadine Belhidane

---

## The task

Given a candidate profile and a job posting, **predict match quality** (0–1).

Use that score to:
- Rank candidates for a job
- Rank jobs for a candidate

Two things matter: **accuracy** (how close is the predicted score?) and **ranking quality** (do the best candidates surface in the top 5?).

<!-- note: This is framed as a regression + ranking problem. The distinction matters — we evaluate with both pointwise metrics (MAE, Spearman) and ranking metrics (NDCG@5). -->

---

## The dataset

**9,460 candidate-job pairs** — 28 job titles × ~341 candidates each

The structure is unusual: **every candidate was scored against every job** (99.1% filled matrix). This is not a real-world sparse matching table — it's a constructed benchmark.

| | |
|---|---|
| `matched_score` | 0–0.97, mean 0.66, **quasi-discrete** |
| Jobs | 28 unique titles |
| Candidates | 341 unique profiles |
| Columns | 35 — text, lists, structured |

The label was computed by a **rule-based algorithm**, not human assessors. Top two values (0.85 and 0.65) account for 29% of all rows.

<!-- note: The quasi-discrete label is key context. It explains why RMSE has an irreducible floor. Mention the floating-point artefacts that reveal the rational number structure. -->

---

## The headline EDA finding

### Candidate skills and job skills share almost no vocabulary

After cleaning and normalisation:

- Only **5.8%** of candidate-job pairs have any token overlap (Jaccard > 0)
- Job `skills_required` uses **97 unique tokens** across 28 jobs
- Candidate `skills` uses **2,797 unique tokens**
- Overlap: **31 tokens**

Job side uses composite phrases as single tokens:
> `'ccna (cisco certified network associate)'` vs candidate's `'ccna'`
> `'asp.net mvc strong understanding of database design'` — one token

**This mismatch is structural. Cleaning does not fix it. It motivates the embedding approach.**

<!-- note: Spend time here. This is the core finding that justifies the whole architecture. The audience needs to understand that this isn't a preprocessing problem — it's a vocabulary design problem in the original data. -->

---

## Other EDA findings

**Experience signal is dominant**
`title_semantic_sim` (JobBERT cosine of candidate titles vs job title): Spearman **+0.25** with score — strongest individual feature
`exp_deficit` (years short of requirement): Spearman **−0.16**

**Career stage matters**
`is_fresher` (detected from career objective text): freshers score **0.56** vs **0.67** for non-freshers — 10-point gap, Spearman −0.16

**Missing-requirement artefact**
21.4% of jobs have no `skills_required`. Mean score for those jobs: **0.73** vs **0.65** when requirements are listed. The algorithm can't penalise skill mismatches if none are specified.

<!-- note: These three findings directly motivated three features we built. Connect each finding to its feature. -->

---

## Feature engineering

**Two text representations per pair:**

`candidate_doc` = career objective + skills (lowercased, deduped) + positions + related_skils_in_job

`job_doc` = job title + skills_required (cleaned, `\n`-split) + responsibilities + educational_requirements

**Structured features (10 total):**

| Feature | Rationale |
|---|---|
| `exp_deficit`, `exp_surplus` | Directional experience gap |
| `title_semantic_sim` | JobBERT cosine: candidate titles vs job title |
| `skill_coverage`, `fuzzy_skill_coverage` | Token + fuzzy overlap |
| `skills_required_count` | Missing-requirement artefact proxy |
| `is_fresher` | Career stage signal |
| `edu_match`, `years_experience` | Standard structured signals |

<!-- note: The skill cleaning step matters — stripped bullets, split compound tokens like 'R or Java' → ['R', 'Java'], removed noise phrases like 'fast typing skill'. -->

---

## Three models

```
data/resume.csv → 02_features.ipynb → features.csv
                                             ↓
                              ┌──────────────┼──────────────┐
                         Model 0         Model A         Model B
                        Heuristic      TF-IDF +       JobBERT +
                       (no training)     Ridge           HGB
```

All models use the **same train/test split**: group-aware by job (80/20), random_state=42. 2,027 test rows across 6 jobs.

<!-- note: The identical split is essential for fair comparison. Emphasise this. -->

---

## Model 0 — Heuristic baseline

**No training.** Hand-crafted weights over 4 signals:

```
score = 0.55 × skill_coverage
      + 0.25 × exp_score (sigmoid of exp_gap)
      + 0.10 × edu_match
      + 0.10 × title_sim (TF-IDF)
```

**Why it fails:** skill_coverage fires for only **6% of pairs** due to the vocabulary mismatch. The formula reduces to an experience + title signal for 94% of the data.

Train Spearman: **0.047** (essentially zero)

| MAE | RMSE | Spearman | NDCG@5 |
|---|---|---|---|
| 0.474 | 0.514 | 0.165 | 0.856 |

<!-- note: The heuristic is not wrong about the signals — it's wrong about the data. skill_coverage is the right idea but the vocabulary gap makes it fire for almost nothing. This is the diagnostic that motivated the approach. -->

---

## Model A — TF-IDF + Ridge

**Skills upweighted 3× in the document** before vectorisation. `ngram_range=(1,2)` captures "machine learning" as a unit.

**Key design choice: job_id one-hot encoding.** EDA found systematic mean score differences across jobs (DBA 0.80, Civil Engineer 0.50) driven partly by missing `skills_required`. Adding 28 job_id dummies lets Ridge learn per-job offsets.

```
[tfidf_cosine, title_semantic_sim, skill_coverage, 
 fuzzy_skill_coverage, exp_deficit, exp_surplus,
 skills_required_count, edu_match, is_fresher, 
 years_experience] + job_id one-hot → Ridge
```

**Ablation:** removing job_id raises Spearman to **0.510** — the text signal provides genuine cross-job discrimination that calibration slightly interferes with.

| MAE | RMSE | Spearman | NDCG@5 |
|---|---|---|---|
| 0.127 | 0.149 | 0.462 | 0.875 |

<!-- note: The intercept is 0.65 — the dataset mean. The model is mostly a per-job offset table with a weak text perturbation. The job_id finding is interesting: removing it actually helps Spearman. For a production system that needs to generalise to new jobs, removing job_id is the right call. -->

---

## Model B — JobBERT + HistGradientBoosting

**Why domain-specific embeddings?** `TechWolf/JobBERT-v3` was pretrained on job postings and professional profiles. It places semantically related professional terms in nearby vector space — handling synonyms and abbreviations that TF-IDF cannot.

**Two additional embedding features:**
- `st_cosine_jobbert` — full document similarity
- `skill_semantic_sim_jobbert` — similarity on skill lists only

```
[st_cosine_jobbert, skill_semantic_sim_jobbert,
 title_semantic_sim, skill_coverage, fuzzy_skill_coverage,
 exp_deficit, exp_surplus, skills_required_count, 
 edu_match, is_fresher, years_experience] + job_id one-hot
 → HistGradientBoosting
```

| | MAE | Spearman | NDCG@5 |
|---|---|---|---|
| B1 MiniLM + Ridge | **0.125** | 0.454 | **0.886** |
| Cross-encoder + Ridge | 0.131 | **0.482** | 0.848 |

<!-- note: HGB instead of Ridge for B2 because tree models handle feature interactions naturally without needing feature scaling. The HGB was more reliant on job_id dummies than Ridge — it used them as primary regression anchors. -->

---

## The key structural insight

The dataset is a **341 × 28 recommendation matrix** (99.1% filled, 9,460 observed pairs). Every candidate was scored against almost every job.

The group-aware split holds out jobs — but every **candidate** appears in both train and test.

**`cand_mean_score`** = candidate's average score across 22 training jobs

This is the latent **"candidate quality"** factor — the scoring algorithm's general assessment of this person. A candidate rated highly across engineering jobs will be rated highly on management jobs too.

| | without `cand_mean_score` | with `cand_mean_score` |
|---|---|---|
| Model A Spearman | 0.462 | **0.645** |
| Cross-encoder NDCG@5 | — | **0.941** |

Single feature. Substantial Spearman gain. This is collaborative filtering applied to a regression problem. (`cand_mean_score` not included in the final saved predictions — exploratory finding.)

**Cold-start limitation:** new candidates with no history cannot use this feature → fall back to content-based model.

<!-- note: This is the "genius" finding. The dataset is not independent samples — it's a recommendation matrix. We were solving the wrong problem (semantic matching) when the dominant signal was candidate quality. This is what you'd say if asked "what did you miss initially?" -->


---

## Results

| | Model 0 | Model A | Model B | Cross-encoder |
|---|---|---|---|---|
| **MAE** | 0.474 | 0.127 | **0.125** | 0.131 |
| **RMSE** | 0.514 | 0.149 | **0.148** | 0.151 |
| **Spearman r** | 0.165 | 0.462 | 0.454 | **0.482** |
| **NDCG@5 (by job)** | 0.856 | 0.875 | **0.886** | 0.848 |

**Cross-encoder wins Spearman (0.482)** — cross-attention sees candidate tokens in job context simultaneously.

**Model B (MiniLM + Ridge) best MAE (0.125) and NDCG@5 (0.886)** — fast at inference, good within-job ranking.

**Adding `cand_mean_score`** substantially boosts Spearman — latent candidate quality is the dominant signal.

<!-- note: The split result is not a contradiction. Model B's HGB with job_id dummies calibrates absolute scores better (lower MAE), which helps within-job ranking. Model A's TF-IDF has more genuine cross-job signal. -->

---

## The vocabulary gap — a concrete example

**Candidate:** embedded software engineer, ML background, C++
**Job:** Head of Internal Control & Compliance — requires audit, inspection, banking

```
Jaccard = 0.000  (zero shared tokens)
```

| | Prediction | Error |
|---|---|---|
| True score | 0.483 | — |
| Model A | 0.363 | **0.121** |
| **Model B** | **0.569** | **0.085** |

**Why Model A misses:** TF-IDF cosine ≈ 0 when Jaccard = 0. No shared tokens → no similarity signal.

**Why Model B is closer:** JobBERT cosine = **0.051** — domain-specific pretraining recognises that analytical/ML skills occupy the same professional space as audit/inspection roles.

This is exactly the vocabulary gap the architecture was designed to bridge.

<!-- note: This is the money slide. Walk through it slowly. The point is not that B is perfect (it still overshoots a bit) — the point is that it sees signal where A is completely blind. -->

---

## Limitations

**The label is rule-based.** Quasi-discrete, computed by an algorithm we can't access. The irreducible RMSE floor is ~half the smallest gap between score levels. Theoretical Spearman ceiling: 0.55–0.60. We reach **0.482** (cross-encoder).

**Candidate leakage.** The group-by-job split prevents job-text leakage but all 341 candidates appear in both train and test. Metrics overestimate generalisation to truly unseen candidates.

**BM25 was tested** — NDCG@5 marginally better (0.898 vs 0.888), but Spearman worse (0.458 vs 0.515). Length normalisation helps within-job ranking but the vocabulary gap affects both approaches equally.

**NDCG@5 is noisy** for tight-distribution jobs. Database Administrator (std = 0.063) contributes near-random NDCG regardless of model quality.

<!-- note: Be honest about the ceiling. 0.515 Spearman is good but not perfect — and the reason isn't model quality, it's label quality. That's actually a positive story: the models are approaching the theoretical limit of what's learnable from this label. -->

---

## Production design

**Two-stage retrieval and re-ranking:**

```
Offline (nightly):
  Encode all profiles with JobBERT-v3 → vector DB (Qdrant/pgvector)

Online — Stage 1 (recall, <50ms):
  Encode query → ANN search → top-50 candidates

Online — Stage 2 (precision, <150ms):
  Compute structured features for top-50
  Run HGB model → re-rank → return top-10
```

**Data flywheel:** recruiter shortlists/rejects → contrastive fine-tuning → model improves as it's used

**ESCO enrichment:** jobs with missing `skills_required` (21.4%) enriched offline by mapping to ESCO occupation → fixes the score inflation artefact found in EDA

End-to-end latency target: **<200ms**

<!-- note: The key insight for the production design is that embedding inference is expensive and should happen offline. The online path is just retrieval + structured feature computation + a small model call. -->

---

## Given more time

**ESCO taxonomy layer** — the right long-term solution. Maps skills, education, and job titles to standardised identifiers. Resolves the vocabulary mismatch structurally. `'nlp'` and `'natural language processing'` become the same concept.

**Fine-tune JobBERT-v3** — contrastive loss on (candidate, job, score) triples makes embeddings respond to this specific matching task rather than general professional similarity.

**NER on `educational_requirements`** — extract degree level and field precisely rather than the ordinal mapping used here.

**`related_skils_in_job` frequency weighting** — this column contains repeated skill mentions. A skill appearing 5× is more prominent than one appearing 1×. We flattened to a unique set; the frequency is real signal.

---

## Discussion

**Defend these choices:**
- Why group-by-job split?
- Why Ridge not XGBoost for Model A?
- Why HistGradientBoosting for Model B?
- Why not fine-tune the transformer?
- What does Spearman 0.515 actually mean in recruiting terms?

**The key number to remember:** NDCG@5 = **0.886** — for any given job, Model B puts the best candidates in the top 5 correctly 89% of the time.

---

*Code: `github.com/[your-repo]`*
*Tech note: `tech_note.md`*
*Production design: `production_design.md`*
