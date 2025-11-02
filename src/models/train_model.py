import os
import yaml
import json
import pickle
import mlflow
import mlflow.sklearn
import mlflow.keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from src.utils.utils import datasplit, sequence_split  # import both
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.models.build_model import ModelBuilder


class ModelTrainer:
    def __init__(self, dataframe, yaml_config_path, target_column):
        """Initialize ModelTrainer with dataframe and configuration."""
        self.df = dataframe.copy()
        self.target_column = target_column
        self.models = {}
        self.histories = {}
        self.metrics = {}

        # --- Load YAML config ---
        if not os.path.exists(yaml_config_path):
            raise FileNotFoundError(f"❌ Config file not found: {yaml_config_path}")

        with open(yaml_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # --- Merge model params if included ---
        model_params_path = self.config.get("include_model_params")
        if model_params_path and os.path.exists(model_params_path):
            with open(model_params_path, "r") as f:
                model_params = yaml.safe_load(f)
                # Merge model params into main config
                self.config.update(model_params)

        # --- Setup MLflow ---
        mlflow_cfg = self.config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "default_experiment"))

        # --- Initialize model builder ---
        self.model_builder = ModelBuilder(config_dict=self.config)

        logger.info("✅ ModelTrainer initialized successfully")

    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
    def _prepare_data(self, model_type="logistic_regression"):
        """
        Prepare and split data depending on model type.
        - For traditional ML models: uses datasplit()
        - For deep learning models: uses sequence_split()
        """
        try:
            logger.info(f"🔄 Preparing data for model type: {model_type}")

            if model_type in ["lstm", "cnn", "cnn_lstm"]:
                (
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    self.tokenizer,
                ) = sequence_split(self.df, self.config)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = datasplit(
                    self.df,
                    test_size=self.config.get("data_preparation", {}).get("test_size", 0.2),
                    random_state=self.config.get("data_preparation", {}).get("random_state", 42),
                )

            logger.info(
                f"✅ Data ready for {model_type} — "
                f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}"
            )

        except Exception as e:
            logger.exception(f"❌ Error preparing data for {model_type}: {str(e)}")
            raise CustomException(e)


    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _get_training_data(self, model_type):
        """Return data for model type"""
        if model_type == "logistic_regression":
            return self.X_train, self.X_test
        else:
            return self.X_train.values, self.X_test.values

    def _calculate_metrics(self, y_true, y_pred, model_name):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        self.metrics[model_name] = metrics
        return metrics

    def _setup_callbacks(self, cfg):
        callbacks = []
        if cfg.get("early_stopping", True):
            callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
        if cfg.get("reduce_lr", True):
            callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3))
        return callbacks

    # -------------------------------------------------------------------------
    # Unified Model Training
    # -------------------------------------------------------------------------
    def train_model(self, model_type, custom_params=None):
        """Generic model training method"""
        with mlflow.start_run(run_name=model_type):
            try:
                logger.info(f"🚀 Starting training for: {model_type.upper()}")

                # Load config
                model_cfg = self.config.get(f"{model_type}_model", {})
                train_cfg = model_cfg.get("training", {})

                # Prepare data
                X_train, X_test = self._get_training_data(model_type)

                # Build model
                model = self.model_builder.build_model(model_type, custom_params)

                # Log model params
                mlflow.log_params({
                    "model_type": model_type,
                    **{k: v for k, v in train_cfg.items() if isinstance(v, (int, float, str, bool))}
                })

                # Train
                if model_type == "logistic_regression":
                    model.fit(X_train, self.y_train)
                    y_pred = model.predict(X_test)
                else:
                    callbacks = self._setup_callbacks(train_cfg.get("callbacks", {}))
                    history = model.fit(
                        X_train, self.y_train,
                        validation_data=(X_test, self.y_test),
                        epochs=train_cfg.get("epochs", 10),
                        batch_size=train_cfg.get("batch_size", 32),
                        callbacks=callbacks,
                        verbose=train_cfg.get("verbose", 1)
                    )
                    self.histories[model_type] = history.history
                    y_pred = (model.predict(X_test) > 0.5).astype("int32")

                # Evaluate
                metrics = self._calculate_metrics(self.y_test, y_pred, model_type)
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)

                # Log model
                if model_type == "logistic_regression":
                    mlflow.sklearn.log_model(model, model_type)
                else:
                    mlflow.keras.log_model(model, model_type)

                self.models[model_type] = model
                logger.info(f"✅ {model_type.upper()} training completed. Accuracy: {metrics['accuracy']:.4f}")

                return model, metrics

            except Exception as e:
                logger.error(f"❌ {model_type.upper()} training failed: {e}")
                raise CustomException(f"{model_type.upper()} training failed: {e}")

    # -------------------------------------------------------------------------
    # Train all
    # -------------------------------------------------------------------------
    def train_all_models(self):
        results = {}
        model_list = self.config.get("models_to_train", ["lstm", "cnn", "cnn_lstm", "logistic_regression"])
        for m in model_list:
            try:
                results[m] = self.train_model(m)
            except Exception as e:
                logger.error(f"Skipping {m}: {e}")
        return results

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    def get_best_model(self):
        if not self.metrics:
            raise ValueError("No models trained yet")
        best = max(self.metrics.items(), key=lambda x: x[1]["accuracy"])
        return best[0], self.models[best[0]], best[1]

    def save_results(self, output_dir="./results"):
        os.makedirs(output_dir, exist_ok=True)
        json.dump(self.metrics, open(os.path.join(output_dir, "metrics.json"), "w"), indent=2)
        pickle.dump(self.histories, open(os.path.join(output_dir, "histories.pkl"), "wb"))

        best_name, best_model, best_metrics = self.get_best_model()
        if best_name == "logistic_regression":
            pickle.dump(best_model, open(os.path.join(output_dir, f"{best_name}.pkl"), "wb"))
        else:
            best_model.save(os.path.join(output_dir, f"{best_name}.h5"))

        logger.info(f"📁 Saved results to {output_dir} — best model: {best_name} (acc={best_metrics['accuracy']:.4f})")
