"""Train borrower/investor classification pipeline."""
from __future__ import annotations
import argparse, json, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from src.data_processing import clean_and_engineer, generate_pseudo_labels, make_model_frame, split_feature_types
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42

def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False))]), categorical_features),
    ], remainder="drop")

def save_bar(items, title, xlabel, path):
    labels, values = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels)*0.28)))
    ax.barh(list(labels)[::-1], list(values)[::-1])
    ax.set_title(title); ax.set_xlabel(xlabel); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def plot_eda(labeled, y, figdir: Path):
    counts = y.value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4)); ax.bar(counts.index, counts.values); ax.set_title("Pseudo-label class balance"); ax.set_ylabel("Record count"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(figdir/"class_balance.png", dpi=160); plt.close(fig)
    for col in ["Bureau_Credit_Score","Debt_to_Income_Ratio","Days_Past_Due (DPD)","Investment_Balance (₦)","Monthly_Net_Income"]:
        if col in labeled.columns:
            vals = pd.to_numeric(labeled[col], errors="coerce").dropna()
            xlabel = col
            if col == "Investment_Balance (₦)": vals, xlabel = np.log1p(vals[vals>=0]), "log1p(Investment Balance)"
            fig, ax = plt.subplots(figsize=(7,4)); ax.hist(vals, bins=35); ax.set_title(f"Distribution: {xlabel}"); ax.set_xlabel(xlabel); ax.set_ylabel("Frequency"); fig.tight_layout()
            safe = col.replace(" ","_").replace("/","_").replace("₦","NGN").replace("(","").replace(")","")
            fig.savefig(figdir/f"distribution_{safe}.png", dpi=160); plt.close(fig)
    numeric = labeled.select_dtypes(include=[np.number])
    if numeric.shape[1] > 2:
        cols = numeric.var(numeric_only=True).sort_values(ascending=False).head(15).index.tolist()
        corr = labeled[cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9,7)); im = ax.imshow(corr, aspect="auto"); ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols))); ax.set_xticklabels(cols, rotation=65, ha="right", fontsize=7); ax.set_yticklabels(cols, fontsize=7); ax.set_title("Correlation heatmap of top numeric features"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout(); fig.savefig(figdir/"correlation_heatmap.png", dpi=160); plt.close(fig)

def evaluate(model, X_test, y_test, classes, figdir: Path):
    pred = model.predict(X_test); proba = model.predict_proba(X_test)
    cm = confusion_matrix(y_test, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8,6)); ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=35, colorbar=False); ax.set_title("Holdout confusion matrix"); fig.tight_layout(); fig.savefig(figdir/"confusion_matrix.png", dpi=160); plt.close(fig)
    y_bin = label_binarize(y_test, classes=classes); roc_auc, pr_auc = {}, {}
    for i, cls in enumerate(classes):
        try:
            roc_auc[cls] = float(roc_auc_score(y_bin[:, i], proba[:, i])); pr_auc[cls] = float(average_precision_score(y_bin[:, i], proba[:, i]))
        except Exception:
            roc_auc[cls] = None; pr_auc[cls] = None
    conf = proba.max(axis=1); correct = (pred == y_test).astype(int); bins = np.linspace(0,1,11); idx = np.digitize(conf, bins)-1; centers=[]; acc=[]; counts=[]
    for b in range(10):
        m = idx == b
        if m.sum(): centers.append(float((bins[b]+bins[b+1])/2)); acc.append(float(correct[m].mean())); counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1], linestyle="--", label="Perfect calibration"); ax.plot(centers, acc, marker="o", label="Model"); ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy"); ax.set_title("Confidence calibration curve"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"calibration_curve.png", dpi=160); plt.close(fig)
    return {"test_macro_f1": float(f1_score(y_test, pred, average="macro")), "test_micro_f1": float(f1_score(y_test, pred, average="micro")), "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0), "report_text": classification_report(y_test, pred, zero_division=0), "roc_auc_ovr": roc_auc, "pr_auc_ovr": pr_auc, "confusion_matrix_labels": list(classes), "confusion_matrix": cm.tolist(), "calibration_bins": {"confidence_bin_center": centers, "observed_accuracy": acc, "count": counts}}

def explain_global(model, classes, figdir: Path, X_sample):
    pre = model.named_steps["preprocess"]; est = model.named_steps["model"]
    try: names = pre.get_feature_names_out().tolist()
    except Exception: names = [f"feature_{i}" for i in range(est.n_features_in_)]
    imp = getattr(est, "feature_importances_", np.zeros(len(names))); top_idx = np.argsort(imp)[::-1][:20]
    global_top = [(names[i], float(imp[i])) for i in top_idx]
    save_bar(global_top, "Top global feature importances", "Tree feature importance", figdir/"global_feature_importance.png")
    class_top = {c: global_top[:10] for c in classes}
    try:
        import shap
        Xt = pre.transform(X_sample); explainer = shap.TreeExplainer(est); sv = explainer.shap_values(Xt)
        mats = sv if isinstance(sv, list) else ([np.asarray(sv)[:, :, i] for i in range(np.asarray(sv).shape[2])] if np.asarray(sv).ndim == 3 else [np.asarray(sv)])
        for ci, cls in enumerate(classes):
            mat = mats[ci] if ci < len(mats) else mats[0]; mean_abs = np.abs(mat).mean(axis=0); idx = np.argsort(mean_abs)[::-1][:10]; class_top[cls] = [(names[i], float(mean_abs[i])) for i in idx]
        all_mean = np.mean([np.abs(m).mean(axis=0) for m in mats[:len(classes)]], axis=0); idx = np.argsort(all_mean)[::-1][:20]
        save_bar([(names[i], float(all_mean[i])) for i in idx], "Top SHAP features: mean absolute impact", "Mean |SHAP value|", figdir/"shap_global_importance.png")
    except Exception as exc: LOGGER.warning("SHAP skipped: %s", exc)
    return {"feature_names": names, "global_top_features": global_top, "class_top_features": class_top}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--sheet", default="REAL DATA"); p.add_argument("--label-column", default=None); p.add_argument("--output-dir", default="artifacts"); p.add_argument("--reports-dir", default="reports"); p.add_argument("--fast", action="store_true")
    args = p.parse_args(); out = Path(args.output_dir); rep = Path(args.reports_dir); figdir = rep/"figures"; out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    path = Path(args.data); raw = pd.read_excel(path, sheet_name=args.sheet) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    cleaned = clean_and_engineer(raw)
    if args.label_column and args.label_column in cleaned.columns: labeled, label_source = cleaned.copy(), "ground_truth"; labeled["segment_label"] = labeled[args.label_column]
    else: labeled, label_source = generate_pseudo_labels(cleaned), "pseudo_labels_rule_based"
    X, y = make_model_frame(labeled), labeled["segment_label"].astype(str); classes = sorted(y.unique().tolist())
    num, cat = split_feature_types(X); plot_eda(labeled, y, figdir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    pre = build_preprocessor(num, cat); cv = StratifiedKFold(n_splits=2 if args.fast else 3, shuffle=True, random_state=RANDOM_STATE)
    baselines = {"LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs"), "RandomForest": RandomForestClassifier(n_estimators=30, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1), "GradientBoosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.08, max_depth=3, random_state=RANDOM_STATE)}
    baseline_results = {}
    for name, est in baselines.items():
        LOGGER.info("Baseline: %s", name); pipe = Pipeline([("preprocess", pre), ("model", est)])
        if args.fast:
            cv_mean = None; cv_std = None
            # Keep fast mode responsive: fit baselines on a stratified subset.
            sub_idx = X_train.groupby(y_train, group_keys=False).apply(lambda g: g.sample(min(len(g), 500), random_state=RANDOM_STATE)).index
            X_fit, y_fit = X_train.loc[sub_idx], y_train.loc[sub_idx]
        else:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
            cv_mean = float(scores.mean()); cv_std = float(scores.std())
            X_fit, y_fit = X_train, y_train
        pipe.fit(X_fit, y_fit); baseline_results[name] = {"cv_macro_f1_mean": cv_mean, "cv_macro_f1_std": cv_std, "holdout_macro_f1": float(f1_score(y_test, pipe.predict(X_test), average="macro"))}
    final_pipe = Pipeline([("preprocess", pre), ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1))])
    param_dist = {"model__n_estimators": [80, 120, 180], "model__max_depth": [None, 8, 12, 18], "model__min_samples_leaf": [1, 3, 5], "model__max_features": ["sqrt", "log2", None]}
    if args.fast:
        best = final_pipe.set_params(model__n_estimators=120, model__max_depth=12, model__min_samples_leaf=3, model__max_features="sqrt")
        best.fit(X_train, y_train)
        best_score = None
        best_params = {"fast_mode": True, "model__n_estimators": 120, "model__max_depth": 12, "model__min_samples_leaf": 3, "model__max_features": "sqrt"}
    else:
        search = RandomizedSearchCV(final_pipe, param_distributions=param_dist, n_iter=8, scoring="f1_macro", cv=cv, n_jobs=1, random_state=RANDOM_STATE)
        search.fit(X_train, y_train); best = search.best_estimator_
        best_score = float(search.best_score_); best_params = search.best_params_
    metrics = evaluate(best, X_test, y_test, classes, figdir); exp = explain_global(best, classes, figdir, X_train.sample(min(120, len(X_train)), random_state=RANDOM_STATE))
    proba = best.predict_proba(X); preds = pd.DataFrame({"entity_id": raw.get("Customer_ID", pd.Series(range(len(raw)))).astype(str), "predicted_class": best.predict(X), "confidence_score": proba.max(axis=1)})
    for i, cls in enumerate(best.classes_): preds[f"probability_{cls}"] = proba[:, i]
    preds.to_csv(out/"predictions.csv", index=False)
    sample = raw.head(1).replace({np.nan: None}).to_dict(orient="records")[0]; (out/"sample_request.json").write_text(json.dumps(sample, indent=2, default=str), encoding="utf-8")
    artifact = {"pipeline": best, "feature_columns": X.columns.tolist(), "numeric_features": num, "categorical_features": cat, "classes": best.classes_.tolist(), "label_source": label_source, "baseline_results": baseline_results, "best_params": best_params, "feature_names": exp["feature_names"], "global_top_features": exp["global_top_features"], "class_top_features": exp["class_top_features"]}
    joblib.dump(artifact, out/"model.joblib")
    all_metrics = {"label_source": label_source, "n_records": int(len(raw)), "n_features": int(X.shape[1]), "class_distribution": y.value_counts().to_dict(), "baseline_results": baseline_results, "best_params": best_params, "best_cv_macro_f1": best_score, **metrics}
    (rep/"metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8"); (rep/"classification_report.txt").write_text(metrics["report_text"], encoding="utf-8")
    LOGGER.info("Saved model to %s; test macro F1 %.4f", out/"model.joblib", metrics["test_macro_f1"])
if __name__ == "__main__": main()
