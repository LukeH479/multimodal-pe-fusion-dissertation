import argparse
import os
import random
import torch
import torch.distributed as dist
import pandas as pd
import torch.nn as nn
import gc
import h5py
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.Base_Models.sklearn_model_wrapper import WrappedSklearnModel
from src.Base_Models.torch_model_wrapper import WrappedTorchModels
from src.Base_Models.thin_threshold_wrapper import ThresholdTunedWrapper

from src.Base_Models.model_2D import PE2DModel
from src.Base_Models.model_2_5D import PE2_5DModel
from src.Base_Models.model_3D import PE3DModel

from src.Base_Models.model_LR import create_LR_model
from src.Base_Models.model_XGBoost import create_XGBoost

from src.CT_ensemble_Models.model_ensembles import create_mean_ensemble
from src.CT_ensemble_Models.model_ensembles import create_LR_ensemble
from src.CT_ensemble_Models.model_ensembles import create_XGB_ensemble
from src.CT_ensemble_Models.model_ensembles import create_meta_estimator_from_ensemble_factory



from src.Training.eval import (
    append_metrics,
    evaluate_binary_classifier,
    evaluate_binary_classifier_predictions,
    compute_meta_shap_summary,
    save_meta_shap_summary,
)
from src.Base_Models.threshold_tuning import select_threshold_constrained, sens_spec_from_probs
from src.Training.oof_artifacts import (
    build_experiment_id,
    build_experiment_dir,
    save_metadata,
    save_fold_predictions,
    save_experiment_predictions,
    load_model_oof_and_test_predictions,
    get_model_artifact_stem,
    get_fold_dir,
)


DEFAULT_CT_METADATA_PATH = Path("datasets/filtered_datasets/ct_metadata.csv")
DEFAULT_EVAL_OUT_DIR = Path(os.environ.get("PE_EVAL_OUT_DIR", "Model_evaluation_metrics"))
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("PE_ARTIFACT_ROOT", "artifacts"))
DATA_ROOT = Path(os.environ.get("PE_DATA_ROOT", "data"))
DEFAULT_OOF_SEED = 42
DEFAULT_SHAP_BACKGROUND_SIZE = 2000
DEFAULT_SHAP_EXPLAIN_SIZE = 2000


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_eval_run_dir(args):
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_id = f"{args.model}_{args.task}_{args.stage}_{job_id}"
    return DEFAULT_EVAL_OUT_DIR / run_id


class SingleModelSpec():
    def __init__(self, model_class, file_name):
        self.model_class = model_class
        self.file_name = file_name

class EnsembleModelSpec():
    def __init__(self, model_class, base_model_names):
        self.model_class = model_class
        self.base_model_names = base_model_names


def _extract_ids_labels(df, target_col):
    ids = df["image_id"].astype(str).tolist()
    labels = pd.to_numeric(df[target_col], errors="raise").astype(int).to_numpy()
    return ids, labels


def _compute_batch_size(model_name, override=None):
    if override is not None:
        return override
    if model_name in ["2D", "LR", "XGB"]:
        return 16
    if model_name == "2.5D":
        return 8
    if model_name == "3D":
        return 6
    return 4


def _stage_exit():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    raise SystemExit(0)


def _build_experiment_and_save_metadata(args, target_col, model_name, extra_metadata):
    experiment_id = build_experiment_id(
        task=args.task,
        target_column=target_col,
        model_name=model_name,
        n_folds=args.oof_folds,
        seed=args.seed,
    )
    metadata = {
        "stage": args.stage,
        "task": args.task,
        "target_column": target_col,
        "model": model_name,
        "n_folds": int(args.oof_folds),
        "seed": int(args.seed),
        **extra_metadata,
    }
    save_metadata(DEFAULT_ARTIFACT_ROOT, experiment_id, metadata)
    return experiment_id


model_map = {
    # Base models train directly from one data source: either CT volumes/slices or EHR features.
    "2D": SingleModelSpec(PE2DModel, "scans2D"),
    "2.5D": SingleModelSpec(PE2_5DModel, "scans2_5D"),
    "3D": SingleModelSpec(PE3DModel, "scans3D"),

    "LR": SingleModelSpec(create_LR_model, "EHR_feature_set"),
    "XGB": SingleModelSpec(create_XGBoost, "EHR_feature_set"),

    "CTMean": EnsembleModelSpec(create_mean_ensemble, ["2D", "2.5D", "3D"]),
    "CTLR": EnsembleModelSpec(create_LR_ensemble, ["2D", "2.5D", "3D"]),
    "CTXGB": EnsembleModelSpec(create_XGB_ensemble, ["2D", "2.5D", "3D"]),

    "FLAT_EHR_LR_META_LR": EnsembleModelSpec (create_LR_ensemble, ["2D", "2.5D", "3D", "LR"]),
    "FLAT_EHR_XGB_META_LR": EnsembleModelSpec(create_LR_ensemble, ["2D", "2.5D", "3D", "XGB"]),
    "FLAT_EHR_LR_META_XGB": EnsembleModelSpec(create_XGB_ensemble, ["2D", "2.5D", "3D", "LR"]),
    "FLAT_EHR_XGB_META_XGB": EnsembleModelSpec(create_XGB_ensemble, ["2D", "2.5D", "3D", "XGB"]),

    "HIERARCHICAL_CTMEAN_EHR_LR_META_LR": EnsembleModelSpec(create_LR_ensemble, ["CTMean", "LR"]),
    "HIERARCHICAL_CTMEAN_EHR_XGB_META_LR": EnsembleModelSpec(create_LR_ensemble, ["CTMean", "XGB"]),
    "HIERARCHICAL_CTLR_EHR_LR_META_LR": EnsembleModelSpec(create_LR_ensemble, ["CTLR", "LR"]),
    "HIERARCHICAL_CTLR_EHR_XGB_META_LR": EnsembleModelSpec(create_LR_ensemble, ["CTLR", "XGB"]),
    "HIERARCHICAL_CTXGB_EHR_LR_META_LR": EnsembleModelSpec(create_LR_ensemble, ["CTXGB", "LR"]),
    "HIERARCHICAL_CTXGB_EHR_XGB_META_LR": EnsembleModelSpec(create_LR_ensemble, ["CTXGB", "XGB"]),
}

BASE_MODELS = {"2D", "2.5D", "3D", "LR", "XGB"}
CT_ENSEMBLES = {"CTMean", "CTLR", "CTXGB"}
FLAT_MODELS = {
    "FLAT_EHR_LR_META_LR",
    "FLAT_EHR_XGB_META_LR",
    "FLAT_EHR_LR_META_XGB",
    "FLAT_EHR_XGB_META_XGB",
}
HIERARCHICAL_MODELS = {
    "HIERARCHICAL_CTMEAN_EHR_LR_META_LR",
    "HIERARCHICAL_CTMEAN_EHR_XGB_META_LR",
    "HIERARCHICAL_CTLR_EHR_LR_META_LR",
    "HIERARCHICAL_CTLR_EHR_XGB_META_LR",
    "HIERARCHICAL_CTXGB_EHR_LR_META_LR",
    "HIERARCHICAL_CTXGB_EHR_XGB_META_LR",
}

ALL_MODELS = BASE_MODELS | CT_ENSEMBLES | FLAT_MODELS | HIERARCHICAL_MODELS




def create_base_model(model_name, device, batch_size=16, tune_threshold=True):
    model_spec = model_map[model_name]
    if model_name in ["LR", "XGB"]:
        if model_name == "XGB":
            model = WrappedSklearnModel(
                model_spec.model_class(),
                model_spec.file_name,
                tune_threshold=tune_threshold,
                validation_split=True,
            )
        else:
            model = WrappedSklearnModel(
                model_spec.model_class(),
                model_spec.file_name,
                tune_threshold=tune_threshold,
            )

    elif model_name in ["2D", "2.5D", "3D"]:
        model = WrappedTorchModels(model_spec.model_class, device, nn.CrossEntropyLoss, torch.optim.AdamW, model_spec.file_name, batch_size=batch_size)

    return model



def create_runtime_model(model_name, device, batch_size):
    if model_name in BASE_MODELS:
        return create_base_model(model_name, device, batch_size)

    if model_name in CT_ENSEMBLES | FLAT_MODELS:
        # Flat stacks take cached predictions from their component models and fit
        # one meta-model directly on top of those prediction columns.
        model_spec = model_map[model_name]
        base_models = [
            create_base_model(base_model_name, device, batch_size, tune_threshold=False)
            for base_model_name in model_spec.base_model_names
        ]
        return ThresholdTunedWrapper(model_spec.model_class(base_models))

    if model_name in HIERARCHICAL_MODELS::
        # HIERARCHICAL stacks first build a CT ensemble branch, then combine that
        # branch with an EHR branch in a second-stage meta-model.
        model_spec = model_map[model_name]
        base_models = []
        for base_model_name in model_spec.base_model_names:
            if base_model_name in CT_ENSEMBLES:
                ensemble_spec = model_map[base_model_name]
                ensemble_base_models = [
                    create_base_model(name, device, batch_size, tune_threshold=False)
                    for name in ensemble_spec.base_model_names
                ]
                ensemble_model = ensemble_spec.model_class(ensemble_base_models)
                base_models.append(ensemble_model)
            else:
                base_models.append(create_base_model(base_model_name, device, batch_size, tune_threshold=False))

        return ThresholdTunedWrapper(model_spec.model_class(base_models))

    raise ValueError(f"Unknown model: {model_name}")



def load_available_image_ids(base_model_name):
    model_spec = model_map[base_model_name]
    data_file = model_spec.file_name

    if base_model_name in {"2D", "2.5D", "3D"}:
        h5_path = DATA_ROOT / f"{data_file}.h5"
        with h5py.File(h5_path, "r") as hf:
            return {img_id.decode("utf-8") for img_id in hf["image_ids"][:]}
    else:
        parquet_path = DATA_ROOT / f"{data_file}.parquet"
        structured_df = pd.read_parquet(parquet_path)
        return set(structured_df.index.astype(str).tolist())




def resolve_task_config(task, target_column, train_split_path, test_split_path):
    if task == "diagnosis":
        target = target_column or "pe_positive"
        train_path = Path(train_split_path) if train_split_path else Path("datasets/filtered_datasets/train_split.csv")
        test_path = Path(test_split_path) if test_split_path else Path("datasets/filtered_datasets/test_split.csv")
    else:
        target = target_column or "12_month_mortality"
        suffix = "prognosis_12m_mortality" if target == "12_month_mortality" else f"prognosis_{target}"
        train_path = Path(train_split_path) if train_split_path else Path(
            f"datasets/filtered_datasets/train_split_{suffix}.csv"
        )
        test_path = Path(test_split_path) if test_split_path else Path(
            f"datasets/filtered_datasets/test_split_{suffix}.csv"
        )

    return target, train_path, test_path



def ensure_target_labels(split_df, target_col, ct_metadata_path):
    if target_col in split_df.columns:
        return split_df

    if not ct_metadata_path.exists():
        raise FileNotFoundError(
            f"Target column {target_col} missing in split and ct metadata not found: {ct_metadata_path}"
        )

    ct_cols = ["image_id", target_col, "pe_positive"]
    ct_df = pd.read_csv(ct_metadata_path, usecols=ct_cols)
    ct_df["image_id"] = ct_df["image_id"].astype(str)

    merged = split_df.merge(ct_df, on="image_id", how="left", validate="1:1")
    return merged



def validate_task_labels(train_df, test_df, task, target_col):
    for split_name, df in [("train", train_df), ("test", test_df)]:
        if target_col not in df.columns:
            raise KeyError(f"Missing target column {target_col} in {split_name} split.")

        target = pd.to_numeric(df[target_col], errors="coerce")
        if target.isna().any():
            raise ValueError(f"Found missing/non-numeric values in {target_col} for {split_name} split.")
        if not target.isin([0, 1]).all():
            raise ValueError(f"{target_col} must be binary 0/1 for {split_name} split.")

        counts = target.value_counts()
        if len(counts) < 2:
            raise ValueError(
                f"{split_name} split has only one class for {target_col}: {counts.to_dict()}"
            )

    if task == "prognosis":
        for split_name, df in [("train", train_df), ("test", test_df)]:
            if "pe_positive" not in df.columns:
                raise KeyError(f"Missing pe_positive in {split_name} split for prognosis validation.")
            pe = pd.to_numeric(df["pe_positive"], errors="coerce")
            if not pe.eq(1).all():
                raise ValueError(f"{split_name} split contains non-PE-positive rows in prognosis mode.")




def run_oof_base_fold_stage(args, device, batch_size, rank_for_logs, target_col, train_df, test_df):
    if args.model not in BASE_MODELS:
        raise ValueError(f"OOF base fold stage supports only base models: {sorted(base_models)}")

    if args.oof_fold_index is None:
        raise ValueError("--oof-fold-index is required when --stage train_oof_base_fold")

    y_train_full = pd.to_numeric(train_df[target_col], errors="raise").astype(int).reset_index(drop=True)
    x_train_full = train_df["image_id"].astype(str).reset_index(drop=True)

    splitter = StratifiedKFold(
        n_splits=args.oof_folds,
        shuffle=True,
        random_state=args.seed,
    )
    folds = list(splitter.split(x_train_full, y_train_full))

    if args.oof_fold_index < 0 or args.oof_fold_index >= len(folds):
        raise ValueError(
            f"Invalid --oof-fold-index={args.oof_fold_index}. Must be in [0, {len(folds)-1}]"
        )

    train_idx, val_idx = folds[args.oof_fold_index]
    x_fold_train = x_train_full.iloc[train_idx]
    y_fold_train = y_train_full.iloc[train_idx]
    x_fold_val = x_train_full.iloc[val_idx]
    y_fold_val = y_train_full.iloc[val_idx]

    # This stage trains one fold only and saves validation/test predictions so
    # stacked models can later be trained from leakage-free OOF features.
    model = create_base_model(args.model, device, batch_size, tune_threshold=False)

    if rank_for_logs == 0:
        print(
            f"OOF fold stage | model={args.model} | fold={args.oof_fold_index}/{args.oof_folds-1} "
            f"| train={len(x_fold_train)} | val={len(x_fold_val)}"
        )

    model.fit(x_fold_train, y_fold_train)

    if rank_for_logs == 0:
        val_prob = model.predict_proba(x_fold_val)
        test_ids = test_df["image_id"].astype(str)
        test_y = pd.to_numeric(test_df[target_col], errors="raise").astype(int)
        test_prob = model.predict_proba(test_ids)

        experiment_id = _build_experiment_and_save_metadata(
            args=args,
            target_col=target_col,
            model_name=args.model,
            extra_metadata={
                "fold_index": int(args.oof_fold_index),
            },
        )

        fold_dir = get_fold_dir(DEFAULT_ARTIFACT_ROOT, experiment_id, args.oof_fold_index)

        if isinstance(model, WrappedSklearnModel):
            model.save_model(fold_dir / "model.joblib")
        elif isinstance(model, WrappedTorchModels):
            model.save_model(get_model_artifact_stem(DEFAULT_ARTIFACT_ROOT, experiment_id, args.oof_fold_index))

        val_path = save_fold_predictions(
            DEFAULT_ARTIFACT_ROOT,
            experiment_id,
            args.oof_fold_index,
            image_ids=x_fold_val.tolist(),
            y_true=y_fold_val.to_numpy(),
            y_prob=val_prob,
            split_name="val",
        )
        test_path = save_fold_predictions(
            DEFAULT_ARTIFACT_ROOT,
            experiment_id,
            args.oof_fold_index,
            image_ids=test_ids.tolist(),
            y_true=test_y.to_numpy(),
            y_prob=test_prob,
            split_name="test",
        )
        print(f"Saved OOF fold artifacts to: {fold_dir}")
        print(f"Saved val predictions: {val_path}")
        print(f"Saved test predictions: {test_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _create_meta_estimator(model_name):
    model_spec = model_map[model_name]
    if not isinstance(model_spec, EnsembleModelSpec):
        raise ValueError(f"Model {model_name} is not an ensemble/meta model.")
    return create_meta_estimator_from_ensemble_factory(model_spec.model_class)


def _resolve_feature_predictions(feature_name, args, target_col, train_ids, test_ids):
    if feature_name == "CTMean":
        # CTMean is reconstructed from the saved 2D/2.5D/3D predictions rather
        # than loaded as a separate precomputed feature file.
        ct_base = ["2D", "2.5D", "3D"]
        train_cols = []
        test_cols = []
        for name in ct_base:
            tr, te = load_model_oof_and_test_predictions(
                artifact_root=DEFAULT_ARTIFACT_ROOT,
                task=args.task,
                target_column=target_col,
                model_name=name,
                n_folds=args.oof_folds,
                seed=args.seed,
                train_image_ids=train_ids,
                test_image_ids=test_ids,
            )
            train_cols.append(tr)
            test_cols.append(te)
        return np.mean(np.column_stack(train_cols), axis=1), np.mean(np.column_stack(test_cols), axis=1)

    return load_model_oof_and_test_predictions(
        artifact_root=DEFAULT_ARTIFACT_ROOT,
        task=args.task,
        target_column=target_col,
        model_name=feature_name,
        n_folds=args.oof_folds,
        seed=args.seed,
        train_image_ids=train_ids,
        test_image_ids=test_ids,
    )


def _build_meta_feature_matrices(args, target_col, train_ids, test_ids, feature_names):
    # Build one column per base model so the stacker sees a clean tabular matrix
    # of prediction features.
    train_cols = []
    test_cols = []
    for feature_name in feature_names:
        tr_pred, te_pred = _resolve_feature_predictions(feature_name, args, target_col, train_ids, test_ids)
        train_cols.append(tr_pred)
        test_cols.append(te_pred)
    return np.column_stack(train_cols), np.column_stack(test_cols)


def run_meta_oof_fold_stage(args, rank_for_logs, target_col, train_df, test_df):
    if args.oof_fold_index is None:
        raise ValueError("--oof-fold-index is required when --stage train_meta_oof_fold")

    model_spec = model_map[args.model]
    if not isinstance(model_spec, EnsembleModelSpec):
        raise ValueError("--stage train_meta_oof_fold requires an ensemble model.")

    estimator = _create_meta_estimator(args.model)
    if estimator is None:
        raise ValueError("train_meta_oof_fold is only for LR/XGB meta estimators. Use train_meta_from_oof for CTMean.")

    feature_names = list(model_spec.base_model_names)
    train_ids, y_train = _extract_ids_labels(train_df, target_col)
    test_ids, y_test = _extract_ids_labels(test_df, target_col)

    X_train_meta, X_test_meta = _build_meta_feature_matrices(
        args=args,
        target_col=target_col,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=feature_names,
    )

    splitter = StratifiedKFold(
        n_splits=args.oof_folds,
        shuffle=True,
        random_state=args.seed,
    )
    folds = list(splitter.split(np.zeros(len(y_train)), y_train))

    if args.oof_fold_index < 0 or args.oof_fold_index >= len(folds):
        raise ValueError(
            f"Invalid --oof-fold-index={args.oof_fold_index}. Must be in [0, {len(folds)-1}]"
        )

    fit_idx, val_idx = folds[args.oof_fold_index]

    # Fit the meta-learner on the in-fold rows only, then score the held-out OOF
    # rows and the external test set.
    estimator.fit(X_train_meta[fit_idx], y_train[fit_idx])

    val_prob = estimator.predict_proba(X_train_meta[val_idx])[:, 1]
    test_prob = estimator.predict_proba(X_test_meta)[:, 1]

    experiment_id = _build_experiment_and_save_metadata(
        args=args,
        target_col=target_col,
        model_name=args.model,
        extra_metadata={
            "feature_models": feature_names,
            "fold_index": int(args.oof_fold_index),
            "fit_rows": int(len(fit_idx)),
            "val_rows": int(len(val_idx)),
        },
    )

    fold_dir = get_fold_dir(DEFAULT_ARTIFACT_ROOT, experiment_id, args.oof_fold_index)

    val_path = save_fold_predictions(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        args.oof_fold_index,
        image_ids=np.asarray(train_ids, dtype=object)[val_idx].tolist(),
        y_true=y_train[val_idx],
        y_prob=val_prob,
        split_name="val",
    )
    test_path = save_fold_predictions(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        args.oof_fold_index,
        image_ids=test_ids,
        y_true=y_test,
        y_prob=test_prob,
        split_name="test",
    )

    joblib.dump(
        {
            "estimator": estimator,
            "feature_models": feature_names,
            "classes": [0, 1],
        },
        fold_dir / "model.joblib",
    )

    if rank_for_logs == 0:
        print(
            f"Meta OOF fold stage | model={args.model} | fold={args.oof_fold_index}/{args.oof_folds-1} "
            f"| fit={len(fit_idx)} | val={len(val_idx)}"
        )
        print(f"Saved meta fold artifacts to: {fold_dir}")
        print(f"Saved val predictions: {val_path}")
        print(f"Saved test predictions: {test_path}")


def run_meta_from_oof_stage(args, rank_for_logs, target_col, train_df, test_df):
    model_spec = model_map[args.model]
    if not isinstance(model_spec, EnsembleModelSpec):
        raise ValueError("--stage train_meta_from_oof requires an ensemble model.")

    feature_names = list(model_spec.base_model_names)
    train_ids, y_train = _extract_ids_labels(train_df, target_col)
    test_ids, y_test = _extract_ids_labels(test_df, target_col)

    X_train_meta, X_test_meta = _build_meta_feature_matrices(
        args=args,
        target_col=target_col,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=feature_names,
    )

    estimator = _create_meta_estimator(args.model)
    if estimator is None:
        # CTMean-style mean ensemble output from cached CT base predictions.
        oof_prob = X_train_meta.mean(axis=1)
        test_prob = X_test_meta.mean(axis=1)
    else:
        # Final stacked model fits once on the full OOF feature matrix, then the
        # selected threshold is stored with the saved artifact.
        estimator.fit(X_train_meta, y_train)
        oof_prob = estimator.predict_proba(X_train_meta)[:, 1]
        test_prob = estimator.predict_proba(X_test_meta)[:, 1]

    threshold, threshold_mode = select_threshold_constrained(y_train, oof_prob)
    test_sens, test_spec = sens_spec_from_probs(y_test, test_prob, threshold)

    experiment_id = _build_experiment_and_save_metadata(
        args=args,
        target_col=target_col,
        model_name=args.model,
        extra_metadata={
            "feature_models": feature_names,
            "threshold": float(threshold),
            "threshold_mode": threshold_mode,
            "test_sensitivity": float(test_sens),
            "test_specificity": float(test_spec),
        },
    )
    exp_dir = build_experiment_dir(DEFAULT_ARTIFACT_ROOT, experiment_id)

    final_train_path = save_experiment_predictions(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        image_ids=train_ids,
        y_true=y_train,
        y_prob=oof_prob,
        split_name="final_train",
    )
    test_path = save_experiment_predictions(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        image_ids=test_ids,
        y_true=y_test,
        y_prob=test_prob,
        split_name="test",
    )

    if estimator is not None:
        model_payload = {
            "estimator": estimator,
            "feature_models": feature_names,
            "threshold": float(threshold),
            "classes": [0, 1],
        }
        joblib.dump(model_payload, exp_dir / "meta_model.joblib")

        if rank_for_logs == 0 and args.run_meta_shap:
            shap_summary = compute_meta_shap_summary(
                estimator,
                X_train=X_train_meta,
                X_test=X_test_meta,
                feature_names=feature_names,
                max_background=DEFAULT_SHAP_BACKGROUND_SIZE,
                max_explain=DEFAULT_SHAP_EXPLAIN_SIZE,
            )
            shap_path = exp_dir / "shap_summary.csv"
            save_meta_shap_summary(shap_summary, shap_path)
            print(f"Saved SHAP summary to: {shap_path}")
            print("Top SHAP meta-features:")
            print(shap_summary.head(20))

    if rank_for_logs == 0:
        eval_out_dir = _get_eval_run_dir(args)
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        plot_prefix = f"{args.model}_{args.task}"
        metrics_out = str(eval_out_dir / "metrics.json")

        evaluate_binary_classifier_predictions(
            y_true=y_test,
            y_prob=test_prob,
            threshold=threshold,
            n_bins=10,
            plots=("roc", "pr", "calibration", "cm", "sens_spec", "decision_curve"),
            target_sensitivity=0.9,
            threshold_sweep_step=0.01,
            metrics_out=metrics_out,
            plots_out_dir=str(eval_out_dir),
            plot_prefix=plot_prefix,
        )

        if estimator is not None and args.run_meta_shap:
            append_metrics(
                {
                    "meta_shap_summary": shap_summary,
                },
                metrics_out,
            )

        print(f"Saved final meta train predictions: {final_train_path}")
        print(f"Saved meta test predictions: {test_path}")
        print(f"Saved metrics to: {metrics_out}")
        if estimator is not None:
            print(f"Saved meta model artifact: {exp_dir / 'meta_model.joblib'}")
        print(
            f"Meta stage completed | model={args.model} | "
            f"test sensitivity={float(test_sens):.4f} | test specificity={float(test_spec):.4f}"
        )


def save_full_base_artifacts(args, model, rank_for_logs, target_col, train_df, test_df, device):
    base_models = {"2D", "2.5D", "3D", "LR", "XGB"}
    if args.model not in base_models or rank_for_logs != 0:
        return None

    train_ids = train_df["image_id"].astype(str).tolist()
    test_ids = test_df["image_id"].astype(str).tolist()
    y_train = pd.to_numeric(train_df[target_col], errors="raise").astype(int).to_numpy()
    y_test = pd.to_numeric(test_df[target_col], errors="raise").astype(int).to_numpy()

    train_prob = model.predict_proba(train_df["image_id"])
    test_prob = model.predict_proba(test_df["image_id"])

    experiment_id = build_experiment_id(
        task=args.task,
        target_column=target_col,
        model_name=args.model,
        n_folds=args.oof_folds,
        seed=args.seed,
    )
    exp_dir = build_experiment_dir(DEFAULT_ARTIFACT_ROOT, experiment_id)

    final_train_path = save_experiment_predictions(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        image_ids=train_ids,
        y_true=y_train,
        y_prob=train_prob,
        split_name="final_train",
    )
    test_path = save_experiment_predictions(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        image_ids=test_ids,
        y_true=y_test,
        y_prob=test_prob,
        split_name="test",
    )

    save_metadata(
        DEFAULT_ARTIFACT_ROOT,
        experiment_id,
        {
            "stage": args.stage,
            "task": args.task,
            "target_column": target_col,
            "model": args.model,
            "n_folds": int(args.oof_folds),
            "seed": int(args.seed),
            "artifact_kind": "full_base_model",
        },
    )

    if isinstance(model, WrappedSklearnModel):
        model_path = exp_dir / "model.joblib"
        model.save_model(model_path)
    elif isinstance(model, WrappedTorchModels):
        model_path = exp_dir / "model"
        model.save_model(model_path)
    else:
        model_path = None

    print(f"Saved full base train predictions: {final_train_path}")
    print(f"Saved full base test predictions: {test_path}")
    if model_path is not None:
        print(f"Saved full base model artifact: {model_path}")

    return np.asarray(test_prob)



def main():
    parser = argparse.ArgumentParser(description="ML pipeline for PE Diagnosis/Prognosis")

    parser.add_argument("--model", type=str, default="LR", choices=sorted(ALL_MODELS),
                        help="Choose the model architecture to use for training and evaluation")
    parser.add_argument("--task", type=str, choices=["diagnosis", "prognosis"], default="diagnosis",
                        help="Task type. Prognosis expects PE-positive cohort and mortality/readmission target.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training. Defaults: 16 for single models, 6 for ensembles, 4 for multimodal stacks.")
    parser.add_argument("--run-meta-shap", action="store_true",
                        help="Run SHAP on fitted stacking meta-learner (LR/XGB only).")
    parser.add_argument("--stage", type=str, default="standard", choices=["standard", "train_oof_base_fold", "train_meta_oof_fold", "train_meta_from_oof"],
                        help="Execution stage. standard runs existing train/eval; train_oof_base_fold trains one base-model fold and saves artifacts; train_meta_oof_fold trains one meta-model fold from cached features; train_meta_from_oof trains final meta model from cached OOF predictions.")
    parser.add_argument("--oof-folds", type=int, default=5,
                        help="Number of stratified folds for OOF base-model training stage.")
    parser.add_argument("--oof-fold-index", type=int, default=None,
                        help="Fold index to train/save in OOF base-model stage (0-based).")
    parser.add_argument("--seed", type=int, default=DEFAULT_OOF_SEED,
                        help="Random seed for OOF fold splits and seed-scoped artifact naming.")
    args = parser.parse_args()
    set_global_seed(args.seed)

    # Resolve the right split files for diagnosis vs prognosis before anything
    # else, because the rest of the pipeline uses the same train/test interface.
    target_col, train_split_path, test_split_path = resolve_task_config(
        args.task,
        target_column=None,
        train_split_path=None,
        test_split_path=None,
    )

    if not train_split_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            f"Split files not found. train={train_split_path}, test={test_split_path}"
        )

    train_df = pd.read_csv(train_split_path)
    test_df  = pd.read_csv(test_split_path)

    train_df["image_id"] = train_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)

    # Some split files do not carry every target column directly, so backfill the
    # target from ct_metadata when needed and then validate the labels.
    ct_metadata_path = DEFAULT_CT_METADATA_PATH
    train_df = ensure_target_labels(train_df, target_col=target_col, ct_metadata_path=ct_metadata_path)
    test_df = ensure_target_labels(test_df, target_col=target_col, ct_metadata_path=ct_metadata_path)
    validate_task_labels(train_df, test_df, task=args.task, target_col=target_col)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DDP if torchrun launched this process
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
        print(f"[Rank {rank}] DDP initialized (world size: {world_size})")

    rank_for_logs = int(os.environ.get("RANK", 0))
    if rank_for_logs == 0:
        print("Running model: " + args.model)
        print(f"Stage: {args.stage}")
        print(f"Task: {args.task} | Target: {target_col}")
        print(f"Seed: {args.seed}")
        print(f"Train split: {train_split_path}")
        print(f"Test split: {test_split_path}")

    batch_size = _compute_batch_size(args.model, override=args.batch_size)
    
    if rank_for_logs == 0:
        print(f"Using batch_size={batch_size} (model type: {args.model})")

    # These stage branches are used when building stacked models from saved OOF
    # artifacts. Each branch exits early after writing its own stage outputs.
    if args.stage == "train_oof_base_fold":
        run_oof_base_fold_stage(
            args=args,
            device=device,
            batch_size=batch_size,
            rank_for_logs=rank_for_logs,
            target_col=target_col,
            train_df=train_df,
            test_df=test_df,
        )
        _stage_exit()

    if args.stage == "train_meta_from_oof":
        run_meta_from_oof_stage(
            args=args,
            rank_for_logs=rank_for_logs,
            target_col=target_col,
            train_df=train_df,
            test_df=test_df,
        )
        _stage_exit()

    if args.stage == "train_meta_oof_fold":
        run_meta_oof_fold_stage(
            args=args,
            rank_for_logs=rank_for_logs,
            target_col=target_col,
            train_df=train_df,
            test_df=test_df,
        )
        _stage_exit()

    # Standard mode trains the requested model end-to-end and writes the usual
    # evaluation plots/metrics into Model_evaluation_metrics.
    model = create_runtime_model(args.model, device, batch_size)


    model.fit(train_df["image_id"], train_df[target_col])

    # For base models we also save prediction artifacts into the shared model
    # artifact store so later ensemble scripts can reuse them.
    cached_test_prob = save_full_base_artifacts(
        args=args,
        model=model,
        rank_for_logs=rank_for_logs,
        target_col=target_col,
        train_df=train_df,
        test_df=test_df,
        device=device,
    )

    metrics_out = None
    shap_summary_out = None
    plots_out_dir = None
    plot_prefix = None
    plots_to_generate = ()
    if rank_for_logs == 0:
        eval_out_dir = _get_eval_run_dir(args)
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        plot_prefix = f"{args.model}_{args.task}"
        plots_out_dir = str(eval_out_dir)
        plots_to_generate = ("roc", "pr", "calibration", "cm", "sens_spec", "decision_curve")

        metrics_out = str(eval_out_dir / "metrics.json")

        if args.run_meta_shap:
            shap_summary_out = str(eval_out_dir / "shap_summary.csv")

    if rank_for_logs == 0 and cached_test_prob is not None:
        # If we already cached test probabilities during artifact export, reuse
        # them here rather than asking the model to score the test set again.
        if hasattr(model, "align_labels_for_available"):
            y_true_eval = np.asarray(model.align_labels_for_available(test_df["image_id"], test_df[target_col]))
        else:
            y_true_eval = np.asarray(
                test_df[target_col].tolist() if hasattr(test_df[target_col], "tolist") else test_df[target_col]
            )

        metrics = evaluate_binary_classifier_predictions(
            y_true=y_true_eval,
            y_prob=cached_test_prob,
            threshold=getattr(model, "threshold", 0.5),
            n_bins=10,
            plots=plots_to_generate,
            target_sensitivity=0.9,
            threshold_sweep_step=0.01,
            metrics_out=metrics_out,
            plots_out_dir=plots_out_dir,
            plot_prefix=plot_prefix if plot_prefix is not None else "eval",
        )
    else:
        metrics = evaluate_binary_classifier(
            model,
            data=test_df["image_id"],
            labels=test_df[target_col],
            device=device,
            threshold=getattr(model, "threshold", 0.5),
            n_bins=10,
            plots=plots_to_generate,
            target_sensitivity=0.9,
            threshold_sweep_step=0.01,
            metrics_out=metrics_out,
            plots_out_dir=plots_out_dir,
            plot_prefix=plot_prefix if plot_prefix is not None else "eval",
        )

    if rank_for_logs == 0 and args.run_meta_shap:
        try:
            shap_summary = compute_meta_shap_summary(
                model,
                X_train=train_df["image_id"],
                X_test=test_df["image_id"],
                max_background=DEFAULT_SHAP_BACKGROUND_SIZE,
                max_explain=DEFAULT_SHAP_EXPLAIN_SIZE,
            )
            if shap_summary_out is not None:
                save_meta_shap_summary(shap_summary, shap_summary_out)
                print(f"Saved SHAP summary to: {shap_summary_out}")
            if metrics_out is not None:
                append_metrics(
                    {
                        "meta_shap_summary": shap_summary,
                    },
                    metrics_out,
                )
            print("Top SHAP meta-features:")
            print(shap_summary.head(20))
        except ValueError as exc:
            print(f"Skipping SHAP: {exc}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
        

    if rank_for_logs == 0:
        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    main()
