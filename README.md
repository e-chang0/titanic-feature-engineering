# Titanic Feature Engineering

Reproducible machine-learning coursework using the real Kaggle Titanic dataset.

Source-code submission: https://github.com/e-chang0/titanic-feature-engineering

This repository contains source code, dependency versions, and a generic report configuration. The personalized PDF, local student information, downloaded data, trained models, and generated experiment results are excluded. Run the commands below to reproduce the analysis and create the artifacts locally.

## Run

Python 3.11 is recommended. In Windows PowerShell:

```powershell
git clone https://github.com/e-chang0/titanic-feature-engineering.git
cd titanic-feature-engineering
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item submission.example.json submission.json
# Edit submission.json locally with your report name and student ID.
python download_data.py
python eda.py
python run_experiments.py --jobs 2 --automl-iterations 30
python make_report.py
python validate_submission.py
```

`submission.json` is ignored by Git. If it is absent, report generation uses the generic `submission.example.json`. Korean PDF generation uses Windows Malgun Gothic; on Linux install NanumGothic or provide `fonts/NanumGothic.ttf` and `fonts/NanumGothicBold.ttf`.

## Assignment coverage

- Real dataset: 891 rows, numeric and categorical predictors; target `Survived`.
- EDA: missing rates, histograms, boxplots, Pearson heatmap, target/categorical countplots, IQR outlier inspection. Exploratory associations use only the training partition.
- Three numeric imputation strategies: mean, median, most frequent. Categorical mode imputation is shared.
- One-hot versus ordinal encoding (integer/label-style feature encoding).
- Standard, MinMax, Robust, and no scaling.
- Six deterministic derived features: FamilySize, IsAlone, FarePerPerson, CabinKnown, Title, AgeGroup.
- Random-forest importance selection of the top 12 transformed columns, fitted within every CV fold; before/after comparison.
- Logistic Regression and Random Forest: 24 configurations/models, identical 5-fold stratified CV splits.
- Five classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC; survival is positive, threshold 0.5, AUC uses probabilities.
- Required Base/Exp-1/Exp-2/Exp-3 combinations and additional one-factor controlled comparisons.

## Bonus components

1. sklearn `Pipeline` and `ColumnTransformer` for fold-local preprocessing and selection.
2. `GridSearchCV`: 6 LR candidates and 12 RF candidates.
3. SHAP beeswarm and waterfall; probability-additivity checks on the frozen RF.
4. Real FLAML AutoML: RF, Extra Trees, and L2 LR, 30 search iterations; independent inner train/validation split.
5. Impurity and permutation importance with 20-repeat SD, plus raw-column permutation through the full feature pipeline.

## Evaluation protocol

The 80/20 stratified split uses random_state=42 (712 training, 179 final test). All preprocessing and supervised selection are fitted inside each training CV fold. Configuration and parameter choices use training data only. AutoML preprocessing is fitted on inner train, never inner validation or final test. All model choices are frozen before final-test metrics.

Base directly uses four complete raw numeric columns (Pclass, SibSp, Parch, Fare), so its comparison includes differences in input information. One-factor ablations isolate preprocessing and derived-feature effects. Primary model selection uses the highest training CV AUC among the two manually tuned models. Test rankings do not trigger reselection.

Selected CV scores have selection optimism and are not nested-CV estimates. Related passengers may cross a random split. SHAP and feature importance describe predictions, not causal effects. Tuning and feature selection do not guarantee improvement; all measured outcomes are retained.

## Files

| File | Purpose |
|---|---|
| `src/titanic_pipeline.py` | Features, preprocessing, selection, CV, tuning, AutoML, evaluation, SHAP |
| `run_experiments.py` | Main experiment entry point |
| `eda.py` | Training-partition EDA and data dictionary |
| `download_data.py` | Public-data download and exact SHA256 verification |
| `make_report.py` | Korean PDF from actual executed CSV/JSON results |
| `validate_submission.py` | Split boundaries, aggregate metrics, saved predictions/models, SHAP, figures, PDF/source-link verification |
| `src/submission_config.py` | Local identity/configuration and report filename |
| `requirements.txt` | Direct dependency versions |
| `requirements-lock.txt` | Full dependency snapshot from the reference run |

## Data provenance

Original dataset description: https://www.kaggle.com/competitions/titanic/data

Downloaded public copy: https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

Expected SHA256: `4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`

The download helper rejects changed source bytes. Data usage conditions remain those of the original dataset/competition. This repository does not grant a new license for the data.
