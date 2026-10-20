# Credit Risk Analysis

An interactive data app that explores a credit dataset and predicts whether a
loan applicant is a good or a bad credit risk. You can filter the data, look at
how the two groups differ, and run the trained model on a row, on a form you
fill in, or on a CSV you upload.

Built for the Information Diamond module at the University of Twente, which
joins databases and machine learning. The point of the project was to show that
the two depend on each other. A model is only as good as the data you prepared
for it.

## Pipeline

The project runs in three stages. Each one writes the input for the next.

```
dataset.csv  ──►  cleaning.py  ──►  query_result.csv
                                          │
                                          ▼
                                     training.py  ──►  model_knn.pkl
                                                       metrics_knn.json
                                                       knn_selection_results.csv
                                          │
                                          ▼
                                       ui.py  (Streamlit app)
```

The model is trained once and saved. The app loads the saved model rather than
retraining, so every user sees the same predictions.

## The app

Three tabs, with filters in the sidebar that apply to all of them.

**Dashboard** shows the active rows, headline figures such as average credit
amount and average loan duration, and a summary of every numeric feature split
by risk class.

**Visuals** has a credit amount histogram, a box plot of loan duration by risk
class, and a correlation heatmap across the numeric features.

**Predict** runs the saved model three ways. You can pick a row from the
filtered table, fill in a form with the main applicant details, or upload a CSV
and download the predictions back. Each prediction shows the class and the
probability behind it.

## Data

1,000 loan applications with 20 features and a good or bad label. Features
cover the loan itself (amount, duration, purpose, instalment rate), the
applicant's finances (checking and savings balances, credit history, other
credit) and their situation (age, employment, housing, job, dependants).

`cleaning.py` does the preparation:

- Removes outliers from every numeric column using an interquartile range filter
- Reports shape, data types, missing values and summary statistics
- One hot encodes every categorical column so the result is fully numeric
- Runs exploratory SQL over the DataFrame with pandasql, comparing average
  credit amount by job, by housing and by personal status

This takes the data from 1,000 rows and 21 columns to 787 rows and 49 columns.

## Model

A K nearest neighbours classifier inside a scikit-learn pipeline, with feature
scaling fitted on the training data only.

Selection searched k over 3, 5, 7, 9 and 11 against uniform and distance
weighting, scored on a held out validation split. The winner was k = 9 with
uniform weighting. Full results are in `knn_selection_results.csv`.

Test set performance, on 158 rows held out from the start:

| Metric | Value |
| --- | --- |
| Accuracy | 0.766 |
| Precision | 0.766 |
| Recall | 0.974 |
| F1 | 0.857 |
| ROC AUC | 0.734 |

## What those numbers actually mean

The F1 of 0.857 looks strong. It is measured on the good class, which is 72% of
the data, so it mostly rewards the model for agreeing that most applicants are
fine.

Here is the test set confusion matrix:

|  | predicted bad | predicted good |
| --- | --- | --- |
| **actually bad** | 10 | 34 |
| **actually good** | 3 | 111 |

The model finds 10 of the 44 bad applicants and approves the other 34. That is a
recall of 0.23 on the class that matters. Always answering "good" would score
72.2% accuracy, and the model scores 76.6%, so the gain over guessing is small.

For credit risk the expensive mistake is lending to someone who does not repay,
not refusing someone who would have. The selection step optimised F1 on the
majority class, which pushes the model the wrong way. An ROC AUC of 0.734 says
the features do carry real signal, so the ranking is usable even though the
default threshold is not. Better next steps would be scoring on the bad class,
weighting the classes, or moving the decision threshold.

## Running it

Needs Python 3.10 or later.

```
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn pandasql
```

The repository already holds a trained model, so you can start the app straight
away:

```
streamlit run ui.py
```

To run the pipeline from the beginning:

```
python cleaning.py     # writes query_result.csv
python training.py     # writes model_knn.pkl and the metrics files
streamlit run ui.py
```

## Files

| File | What it is |
| --- | --- |
| `dataset.csv` | the raw data |
| `cleaning.py` | cleaning, encoding and exploratory SQL |
| `query_result.csv` | the cleaned and encoded data, generated |
| `training.py` | model selection, training and evaluation |
| `model_knn.pkl` | the saved pipeline, generated |
| `metrics_knn.json` | test metrics, generated |
| `knn_selection_results.csv` | every parameter combination tried, generated |
| `Model_Evaluation_Report_KNN.md` | the evaluation report, generated |
| `ui.py` | the Streamlit app |

## Notes

The outlier filter drops 213 of 1,000 rows, which is 21% of the data. Some of
those are probably genuine large loans rather than errors, and removing them
narrows what the model ever sees.

`cleaning.py` calls `pd.get_dummies(dataset, drop_first=2)`. That argument takes
a boolean, and 2 is treated as true, so it works by accident.

`model_knn.pkl` was saved with scikit-learn 1.7.2. Loading it under a different
version warns and is not guaranteed to behave the same. Rerun `training.py` if
you hit that.

Built as a team project.
