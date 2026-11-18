# Predicting student depression from lifestyle and academic factors

An end-to-end binary classification pipeline that predicts whether a university student reports depression from lifestyle, academic and personal variables, and identifies which factors are most strongly associated with it.

Course project for *Machine Learning*, built in Python with scikit-learn and XGBoost. The whole analysis (cleaning, preprocessing, model selection, bias-variance analysis, feature importance) is in the notebook, with the reasoning for each choice written next to the code.

**Research question:** which lifestyle, academic and personal factors are most strongly associated with depression in university students?

> This is a course project on a public dataset. It is not a diagnostic tool, and its results describe statistical associations, not causes.

## Data

- **Source:** [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Kaggle), 27,901 observations and 18 columns.
- **Target:** `Depression` (0 = not depressed, 1 = depressed). The classes are mildly imbalanced (58.6% vs 41.4%), so I used class weighting where useful instead of resampling such as SMOTE.
- **Features:** academic pressure, study satisfaction, CGPA, work/study hours, sleep duration, dietary habits, financial stress, family history, gender, age, and a question on suicidal thoughts.

## Approach

**Cleaning.**
- Removed 30 rows (0.11%) with an uninterpretable "Others" value in Sleep Duration or Dietary Habits.
- Dropped `id` and the constant or near-constant columns (`Profession`, `Work Pressure`, `Job Satisfaction`), and the high-cardinality `City` and `Degree`, leaving 11 features.
- Only 3 missing values (Financial Stress), handled inside the pipeline.

**Preprocessing.** A single `ColumnTransformer` with one pipeline per feature type:
- Numerical features: mean imputation and standard scaling.
- Ordinal features (Sleep Duration, Dietary Habits): ordinal encoding with an explicit category order, then scaling.
- Binary features (suicidal thoughts, family history, gender): ordinal encoding.

Everything is fit inside the cross-validation folds, so no information leaks from validation data.

**Split.** Stratified 80/20 train/test split (22,296 / 5,575), fixed seed (42).

**Model selection.** Nested cross-validation on the training set: the inner loop (`RandomizedSearchCV`, 3-fold, 40 iterations, F1) tunes hyperparameters and the outer loop (5-fold) estimates generalization. Four model families were compared: Logistic Regression, Decision Tree, Random Forest and XGBoost.

- Mean outer F1: **0.871 ± 0.002**.
- XGBoost won 3 of 5 outer folds and Logistic Regression 2, with practically identical scores.
- I chose **Logistic Regression** because performance was equivalent and its coefficients are directly interpretable, which matters for a project that aims to understand the factors and not only to predict.

**Refinement.** A second search for the regularization strength and penalty (`RepeatedStratifiedKFold`, 5 folds × 3 repeats) selected an L2 penalty with C ≈ 0.54 (CV F1 0.872).

## Results (held-out test set)

| Metric | Value |
| --- | --- |
| F1-score | 0.8695 |
| AUC-ROC | 0.918 |
| Accuracy | 0.85 |
| Precision / recall (depressed) | 0.86 / 0.88 |
| Precision / recall (not depressed) | 0.83 / 0.79 |

The confusion matrix shows 389 false negatives and 475 false positives out of 5,575 students.

**Bias-variance analysis.** Training and validation F1 converge quickly at about 0.87 in the learning curve, and the validation curve is flat for C ≥ 0.01. The model shows low variance and no sign of overfitting.

## Feature importance

Two methods (absolute coefficients and permutation importance on the test set) give a consistent ranking:

| Feature | Permutation importance (drop in F1) | Depression rate in the data |
| --- | --- | --- |
| Suicidal thoughts | 0.089 | 23% (no) vs 79% (yes) |
| Academic pressure | 0.066 | 19% (level 1) to 86% (level 5) |
| Financial stress | 0.029 | 32% (level 1) to 81% (level 5) |
| Age | 0.016 | – |
| Work/study hours | 0.010 | – |

Gender has near-zero importance: the depression rate is almost identical for female (58%) and male (59%) students.

## Limitations

- **Association, not causation.** The data are observational and cross-sectional.
- **Circularity.** Suicidal ideation is a symptom of depression, not an independent risk factor, so including it makes the prediction partly circular. The remaining top features (academic pressure, financial stress, sleep, diet) are the ones that could inform prevention.
- **Academic pressure level 0.** It shows a 44% depression rate, which probably reflects missing or non-applicable values rather than a real pressure level.
- **Test set use.** The test set was also used once to compare the five nested-CV winners before choosing the final model. The two finalists differed by less than 0.003 F1, so the impact is small, but a stricter design would choose the model from cross-validation scores only.
- **Self-reported data.** The dataset comes from a public Kaggle upload and I could not verify how it was collected.

## How to run

The notebook was run on Google Colab (Python 3.13). To run it:

1. Download the dataset from the Kaggle link above and make it available as `Student Depression Dataset.csv`, at the path used in the notebook (`/Student Depression Dataset.csv`).
2. Install the dependencies:

```bash
pip install pandas numpy matplotlib scipy scikit-learn missingno==0.5.2 xgboost==3.2.0
```

3. Run all cells. The nested cross-validation is the slowest step (about 12 minutes on Colab).
