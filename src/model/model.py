import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    make_scorer,
    roc_auc_score,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.svm import (
    SVC,
)

from sklearn.multioutput import (
    MultiOutputClassifier,
)

from scipy.sparse import (
    load_npz,
    hstack
)
import joblib

from utils.io import (
    read_csv,
    read_excel,
    write_csv,
    save_figure
)

DATA_DIR = "../data"
OUT_DIR = "./out/models"
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

source_type = "tfidf"
ngrams_type = "1-2gram"
prefix = f"{source_type}_{ngrams_type}"
use_efeatures = True

articles_adu = read_excel("OpArticles_ADUs.xlsx", directory=DATA_DIR)

def get_value_type(label):
    if label == "Value(-)":
        return 0
    if label == "Value(+)":
        return 2
    return 1

def get_label(label_value):
    label = label_value["label"]
    value = label_value["value_type"]

    if label == 2:
        return label + value
    else:
        return label

def get_label_name(label):
    if label == 0:
        return "Fact"
    if label == 1:
        return "Policy"
    if label == 2:
        return "Value(-)"
    if label == 3:
        return "Value"
    if label == 4:
        return "Value(+)"

# vectorizer = joblib.load(f"{prefix}_vectorizer.joblib")
feature_matrix = load_npz(f"{prefix}_feature_matrix.mat.npz")
if use_efeatures:
    efeatures = read_csv("efeatures.csv")
    efeatures.drop(columns=["label", "value_type"], inplace=True)
    efeatures["polarity"] = efeatures[["adu_polarity", "token_polarity", "blob_polarity"]].mean(axis=1)


y = articles_adu[["label"]].copy()
y["value_type"] = y["label"].apply(get_value_type)
y.loc[y["label"].str.startswith("Value"), "label"] = "Value"

le = LabelEncoder()
y["label"] = le.fit_transform(y["label"])

if use_efeatures:
    X = hstack([
        feature_matrix,
        efeatures[["n_entities", "blob_subjectivity", "polarity"]].to_numpy()
    ])

else:
    X = feature_matrix.copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=SEED,
    test_size=0.20
)

y_train_single = y_train.copy()
y_train_single["label"] = y_train_single.apply(get_label, axis=1)
y_train_single.drop(columns=["value_type"], inplace=True)

y_test_single = y_test.copy()
y_test_single["label"] = y_test_single.apply(get_label, axis=1)
y_test_single.drop(columns=["value_type"], inplace=True)

# Model
K_FOLDS = 5

def tune_random_forest(X, y) -> GridSearchCV:
    hyperparam_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [50, None],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    print("\n --- Tuning RandomForest --- \n")

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=SEED,
        ),
        param_grid=hyperparam_grid,
        scoring=make_scorer(f1_score, greater_is_better=True, average="weighted"),
        cv=5,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return grid_search

def tune_logistic_regression(X, y) -> GridSearchCV:
    hyperparam_grid = {
        "penalty": ["l2", "l1"],
        "C": [0.5, 1.0],
        "solver": ["liblinear", "saga"],
        "class_weight": ["balanced", None]
    }

    print("\n --- Tuning LogisticRegression --- \n")

    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=SEED),
        param_grid=hyperparam_grid,
        scoring=make_scorer(f1_score, greater_is_better=True, average="weighted"),
        cv=5,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return grid_search

def tune_gradient_boosting(X, y) -> GridSearchCV:
    hyperparam_grid = {
        "max_depth": [3, 5, None],
        "max_features": ["sqrt", "log2"],
        "learning_rate": [0.1, 0.2],
        "loss": ["deviance", "exponential"]
    }

    print("\n --- Tuning GradientBoosting --- \n")

    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=SEED),
        param_grid=hyperparam_grid,
        scoring=make_scorer(f1_score, greater_is_better=True, average="weighted"),
        cv=5,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return grid_search

def tune_SVC(X, y) -> GridSearchCV:
    hyperparam_grid = {
        "C": [0.5, 1.0, 1.5],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    }

    print("\n --- Tuning SVC --- \n")

    grid_search = GridSearchCV(
        estimator=SVC(random_state=SEED),
        param_grid=hyperparam_grid,
        scoring=make_scorer(f1_score, greater_is_better=True, average="weighted"),
        cv=5,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return grid_search

PRE_COMPUTED = True
if not PRE_COMPUTED:
    rf_tune = tune_random_forest(X_train, y_train_single.values.ravel())
    print("\n --- Best Params RandomForest --- \n")
    print(rf_tune.best_params_)
    lr_tune = tune_logistic_regression(X_train, y_train_single.values.ravel())
    print("\n --- Best Params LogisticRegression --- \n")
    print(lr_tune.best_params_)
    svc_tune = tune_SVC(X_train, y_train_single.values.ravel())
    print("\n --- Best Params SVC --- \n")
    print(svc_tune.best_params_)
    gb_tune = tune_gradient_boosting(X_train, y_train_single.values.ravel())
    print("\n --- Best Params GradientBoosting --- \n")
    print(gb_tune.best_params_)
    
    rf = RandomForestClassifier(
        random_state=SEED,
        **rf_tune.best_params_
    )

    lr = LogisticRegression(
        random_state=SEED,
        **lr_tune.best_params_,
    )

    svc = SVC(
        random_state=SEED,
        **svc_tune.best_params_,
    )

    gb = GradientBoostingClassifier(
        random_state=SEED,
        **gb_tune.best_params_,
    )

    momc=MultiOutputClassifier(
        estimator=GradientBoostingClassifier(
            random_state=SEED,
            **gb_tune.best_params_,
        ),
    )

else:
    rf = RandomForestClassifier(
        random_state=SEED,
        class_weight="balanced_subsample",
        criterion="gini",
        max_depth=50,
        max_features="sqrt",
    )

    lr = LogisticRegression(
        random_state=SEED,
        C=1.0,
        class_weight=None,
        penalty="l2",
        solver="liblinear",
    )

    svc = SVC(
        random_state=SEED,
        C=1.5,
        gamma="scale",
        kernel="linear",
        probability=True
    )

    gb = GradientBoostingClassifier(
        random_state=SEED,
        learning_rate=0.2,
        loss="deviance",
        max_depth=50,
        max_features="log2"
    )

    momc = MultiOutputClassifier(
        estimator=GradientBoostingClassifier(
            random_state=SEED,
            learning_rate=0.2,
            loss="deviance",
            max_depth=50,
            max_features="log2"
        ),
    )

# Train models
for name, model in zip(
        ["RandomForest", "LogisticRegression", "GradientBoosting", "SVC"],
        [rf, lr, gb, svc]
    ):

    print(f"Fitting {name} model\n")

    model.fit(X_train, y_train_single.values.ravel())

print("Fitting multioutput model\n")
momc.fit(X_train, y_train)

joblib.dump(rf, Path(OUT_DIR, f"{prefix}_random_forest.joblib"))
joblib.dump(lr, Path(OUT_DIR, f"{prefix}_logistic_regression.joblib"))
joblib.dump(svc, Path(OUT_DIR, f"{prefix}_svc.joblib"))
joblib.dump(gb, Path(OUT_DIR, f"{prefix}_gradient_boosting.joblib"))
joblib.dump(momc, Path(OUT_DIR, f"{prefix}_multioutput.joblib"))

classes = ["Fact", "Policy", "Value(-)", "Value", "Value(+)"]
cols = [f"{c}_{col}" for c in classes for col in ["fpr", "tpr", "auc"]]
cols = ["classifier"]+cols

result_table = pd.DataFrame(columns=cols)
report = pd.DataFrame()

print("Building results\n")

for name, model in zip(
        ["RandomForest", "LogisticRegression", "GradientBoosting", "SVC"],
        [rf, lr, gb, svc]
    ):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    row = {
        "classifier": name,
    }

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test_single,
        y_pred,
        display_labels=classes,
        ax=ax
    )

    ax.grid(False)
    ax.set_title(f"{name} - Confusion Matrix")
    save_figure(fig, f"{prefix}_{name}_confusion.png", directory=OUT_DIR, format="png", dpi=150)
    plt.clf()

    aux = classification_report(y_test_single, y_pred, labels=[0, 1, 2, 3, 4], target_names=classes, output_dict=True)
    aux = pd.DataFrame(data=aux).reset_index()
    aux["classifier"] = name

    report = pd.concat([report, aux], ignore_index=True)

    for i, c in enumerate(classes):
        y_t = y_test_single["label"].apply(lambda x: 1 if x == i else 0)
        y_p = y_prob[:, i]
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)

        row[f"{c}_fpr"] = fpr
        row[f"{c}_tpr"] = tpr
        row[f"{c}_auc"] = auc
    
    result_table.loc[len(result_table.index)] = row

y_pred = momc.predict(X_test)
y_prob = momc.predict_proba(X_test)

y_pred_t = np.apply_along_axis(lambda row: row[0] + row[1] if row[0] == 2 else row[0], 1, y_pred)

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test_single,
    y_pred_t,
    display_labels=classes,
    ax=ax
)

ax.grid(False)
ax.set_title("Multioutput - Confusion Matrix")
save_figure(fig, f"{prefix}_multioutput_confusion.png", directory=OUT_DIR, format="png", dpi=150)
plt.clf()

aux = classification_report(y_test_single, y_pred_t, labels=[0, 1, 2, 3, 4], target_names=classes, output_dict=True)
aux = pd.DataFrame(data=aux).reset_index()
aux["classifier"] = "Multioutput"

report = pd.concat([report, aux], ignore_index=True)

y_prob_t = np.zeros(shape=(y_prob[0].shape[0], 5))

for i in range(y_prob_t.shape[0]):
    f_prob, p_prob, v_prob = y_prob[0][i]
    neg_prob, neu_prob, pos_prob = y_prob[1][i]

    y_prob_t[i] = np.array([f_prob, p_prob, v_prob*neg_prob, v_prob*neu_prob, v_prob*pos_prob])

row = {
    "classifier": "Multioutput",
}

for i, c in enumerate(classes):
    y_t = y_test_single["label"].apply(lambda x: 1 if x == i else 0)
    y_p = y_prob_t[:, i]
    fpr, tpr, _ = roc_curve(y_t, y_p)
    auc = roc_auc_score(y_t, y_p)

    row[f"{c}_fpr"] = fpr
    row[f"{c}_tpr"] = tpr
    row[f"{c}_auc"] = auc

result_table.loc[len(result_table.index)] = row

result_table.set_index("classifier", inplace=True)

report = report[~(report["index"] == "support")].set_index(["classifier", "index"])

write_csv(
    report,
    f"{prefix}_report.csv",
    OUT_DIR
)

fig, axs = plt.subplots(figsize=(30, 12), ncols=5)

for i, c in enumerate(classes):
    ax = axs[i]
    for classifier in result_table.index:
        ax.plot(result_table.loc[classifier][f"{c}_fpr"],
                    result_table.loc[classifier][f"{c}_tpr"],
                    label=f"{classifier}, AUC={result_table.loc[classifier][f'{c}_auc']:.3f}"
        )
    ax.plot([0,1], [0,1], color='orange', linestyle='--')

    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("False Positive Rate", fontsize=15)

    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate", fontsize=15)

    ax.set_title(f'ROC Curve - {c}', fontweight='bold', fontsize=15)
    ax.legend(prop={'size':10}, loc='lower right')

save_figure(fig, f"{prefix}_roc.png", directory=OUT_DIR, format="png", dpi=150)