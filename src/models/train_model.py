"""
Model Training Module for Customer Review Sentiment Analysis Pipeline

This module provides comprehensive training capabilities for multiple machine learning models.
It provides functionality to:
    - Prepare training and test data for different model architectures:
        * Vector-based data for traditional ML models (Logistic Regression)
        * Sequence-based data for deep learning models (LSTM, CNN, CNN-LSTM)
    - Train multiple model types with flexible configuration:
        * Logistic Regression for baseline performance
        * LSTM models for sequence-based learning
        * CNN models for pattern recognition
        * Hybrid CNN-LSTM models for combined approaches
    - Configure and apply training callbacks:
        * EarlyStopping to prevent overfitting
        * ReduceLROnPlateau for adaptive learning rate adjustment
    - Track experiments and log metrics using MLflow:
        * Log model parameters and hyperparameters
        * Log trained models for version control
        * Track training history and performance metrics
    - Save trained models in appropriate formats:
        * .pkl format for scikit-learn models (Logistic Regression)
        * .h5 format for Keras/TensorFlow models (LSTM, CNN, CNN-LSTM)
    - Support batch training of multiple models with a single method call

The ModelTrainer class orchestrates the complete training workflow, reading configurations from
YAML files, preparing data appropriately for each model type, training with optimal settings,
and persisting both models and training artifacts. The module integrates with MLflow for
comprehensive experiment tracking and includes robust error handling and logging throughout.
"""

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
        self.df = dataframe.copy()
        self.target_column = target_column
        self.models = {}
        self.histories = {}

        #  Load YAML config
        if not os.path.exists(yaml_config_path):
            raise FileNotFoundError(f"Config file not found: {yaml_config_path}")
        with open(yaml_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        #  Merge model params if included
        model_params_path = self.config.get("include_model_params")
        if model_params_path and os.path.exists(model_params_path):
            with open(model_params_path, "r") as f:
                model_params = yaml.safe_load(f)
                self.config.update(model_params)

        #  Setup MLflow
        mlflow_cfg = self.config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "default_experiment"))

        #   Initialize model builder
        self.model_builder = ModelBuilder(config_dict=self.config)

        logger.info("ModelTrainer initialized successfully")

    #                         -
    def _prepare_data(self, model_type="logistic_regression"):
        try:
            logger.info(f"Preparing data for model type: {model_type}")
            if model_type in ["lstm", "cnn", "cnn_lstm"]:
                self.X_train, self.X_test, self.y_train, self.y_test, self.tokenizer = (
                    sequence_split(self.df, self.config)
                )
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = datasplit(
                    self.df,
                    test_size=self.config.get("data_preparation", {}).get(
                        "test_size", 0.2
                    ),
                    random_state=self.config.get("data_preparation", {}).get(
                        "random_state", 42
                    ),
                )
            logger.info(
                f"Data ready for {model_type} — Train: {self.X_train.shape}, Test: {self.X_test.shape}"
            )
        except Exception as e:
            logger.exception(f"Error preparing data for {model_type}: {str(e)}")
            raise CustomException(e)

    def _get_training_data(self, model_type):
        self._prepare_data(model_type=model_type)
        return (
            (self.X_train, self.X_test)
            if model_type == "logistic_regression"
            else (self.X_train, self.X_test)
        )

    def _setup_callbacks(self, cfg):
        callbacks = []
        if cfg.get("early_stopping", True):
            callbacks.append(
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            )
        if cfg.get("reduce_lr", True):
            callbacks.append(
                ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
            )
        return callbacks

    def train_model(self, model_type, custom_params=None):
        """Generic model training method"""
        with mlflow.start_run(run_name=model_type):
            try:
                logger.info(f"Training: {model_type.upper()}")

                model_cfg = self.config.get(f"{model_type}_model", {})
                train_cfg = model_cfg.get("training", {})

                X_train, X_test = self._get_training_data(model_type)
                model = self.model_builder.build_model(model_type, custom_params)

                mlflow.log_params(
                    {
                        "model_type": model_type,
                        **{
                            k: v
                            for k, v in train_cfg.items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    }
                )

                if model_type == "logistic_regression":
                    model.fit(X_train, self.y_train)
                    logger.info("logistic regression model trained")
                else:
                    callbacks = self._setup_callbacks(train_cfg.get("callbacks", {}))
                    history = model.fit(
                        X_train,
                        self.y_train,
                        validation_data=(X_test, self.y_test),
                        epochs=train_cfg.get("epochs", 10),
                        batch_size=train_cfg.get("batch_size", 32),
                        callbacks=callbacks,
                        verbose=train_cfg.get("verbose", 1),
                    )
                    self.histories[model_type] = history.history

                # Save trained model
                output_dir = os.path.join("artifacts", "models", model_type)
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(
                    output_dir,
                    (
                        f"{model_type}.pkl"
                        if model_type == "logistic_regression"
                        else f"{model_type}.h5"
                    ),
                )

                if model_type == "logistic_regression":
                    pickle.dump(model, open(model_path, "wb"))
                    mlflow.sklearn.log_model(model, model_type)
                else:
                    model.save(model_path)
                    mlflow.keras.log_model(model, model_type)

                self.models[model_type] = model
                logger.info(f"Saved model to {model_path}")
                return model

            except Exception as e:
                logger.error(f"Training failed for {model_type}: {e}")
                raise CustomException(f"{model_type.upper()} training failed: {e}")

    def train_all_models(self):
        """Train all models defined in config"""
        results = {}
        for m in self.config.get(
            "models_to_train", ["logistic_regression", "lstm", "cnn", "cnn_lstm"]
        ):
            try:
                model = self.train_model(m)
                results[m] = model
            except Exception as e:
                logger.error(f"Skipping {m}: {e}")
        return results
