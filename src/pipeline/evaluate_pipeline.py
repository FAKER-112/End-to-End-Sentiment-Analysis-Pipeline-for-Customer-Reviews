"""
Evaluation Pipeline Module for Customer Review Sentiment Analysis

This module provides comprehensive evaluation capabilities for comparing and selecting the best model.
It provides functionality to:
    - Load or prepare cleaned data for evaluation:
        * Load pre-processed data from disk if available
        * Automatically run data cleaning pipeline if data is missing
    - Load trained models from disk with flexible options:
        * Load all models from artifacts directory for comparison
        * Load a specific selected model for targeted evaluation
        * Automatically train models if artifacts are missing (configurable)
        * Force retrain all models from scratch (configurable)
    - Support multiple model formats:
        * .h5 format for Keras/TensorFlow deep learning models
        * .pkl format for scikit-learn traditional ML models
    - Prepare appropriate test data for different model types:
        * Sequence-based test data for deep learning models (LSTM, CNN, CNN-LSTM)
        * Vector-based test data for traditional ML models (Logistic Regression)
    - Evaluate all loaded models using consistent metrics:
        * Accuracy, Precision, Recall, F1 Score (weighted averages)
    - Compare models and identify the best performer based on accuracy
    - Save comprehensive evaluation results:
        * Metrics for all evaluated models in JSON format
        * Best-performing model for production deployment
    - Provide flexible configuration via YAML files for:
        * Data source paths and preprocessing options
        * Model loading strategies and training fallbacks
        * Evaluation output directories

The EvaluationPipeline class orchestrates the complete evaluation workflow, from data preparation
through model comparison to final model selection. It intelligently handles missing data or models
by falling back to data cleaning or model training as configured. The module is designed for
both standalone evaluation runs and integration into larger ML workflows.
"""

import os, sys
import sys
import yaml
import json
import pickle
import numpy as np

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from tensorflow.keras.models import load_model
from src.data.clean_data import CleanData
from src.data.load_data import LoadData
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.logger import logger
from src.utils.exception import CustomException

# Ensure root path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)


class EvaluationPipeline:
    """End-to-end pipeline for model evaluation and selection."""

    def __init__(self, pipeline_config_path="configs/evaluation_pipeline.yaml"):
        try:
            logger.info("🔧 Initializing Evaluation Pipeline...")
            if not os.path.exists(pipeline_config_path):
                raise FileNotFoundError(f"Config not found: {pipeline_config_path}")

            with open(pipeline_config_path, "r") as f:
                self.pipeline_config = yaml.safe_load(f).get("evaluation_pipeline", {})

            # Parse config sections
            self.data_cfg = self.pipeline_config.get("data", {})
            self.model_cfg = self.pipeline_config.get("models", {})
            self.eval_cfg = self.pipeline_config.get("evaluation", {})
            self.train_cfg = self.pipeline_config.get("training", {})

            logger.info(f"✅ Loaded evaluation config from {pipeline_config_path}")
        except Exception as e:
            logger.error("❌ Failed to initialize EvaluationPipeline.")
            raise CustomException(e)

    # -------------------------------------------------------------------------
    def _load_or_prepare_data(self):
        """Load preprocessed data or generate cleaned data if missing."""
        try:
            cleaned_path = self.data_cfg.get(
                "cleaned_data_path", "data/processed/cleaned_data.csv"
            )

            if os.path.exists(cleaned_path):
                logger.info(f"📂 Loading cleaned data from {cleaned_path}")
                import pandas as pd

                df = pd.read_csv(cleaned_path)
            else:
                logger.warning("⚠️ Cleaned data not found, running CleanData()...")
                df = CleanData().clean_data()

            logger.info(f"✅ Data ready for evaluation: {df.shape}")
            return df

        except Exception as e:
            logger.error("❌ Failed to load or prepare data.")
            raise CustomException(e)

    def _load_models(self, df=None):
        """Load models or train depending on config flags."""
        source_dir = self.model_cfg.get("source_dir", "artifacts/models")
        evaluate_all = self.model_cfg.get("evaluate_all", True)
        selected_model = self.model_cfg.get("selected_model", None)
        train_if_missing = self.model_cfg.get("train_models_if_missing", False)
        train_all = self.model_cfg.get("train_all_models", False)

        models = {}

        try:
            # ✅ 1. Force retrain all models if specified
            if train_all:
                logger.info(
                    "🧠 Config specifies train_all_models=True — retraining all models..."
                )
                trainer = ModelTrainer(
                    dataframe=df,
                    yaml_config_path=self.train_cfg.get("config_path"),
                    target_column=self.train_cfg.get("target_column"),
                )
                models = trainer.train_all_models()
                logger.info("✅ All models retrained and loaded successfully.")
                return models

            # ✅ 2. Load from artifacts/models
            if not os.path.exists(source_dir) or not any(os.scandir(source_dir)):
                if train_if_missing:
                    logger.warning("⚠️ No models found, training all models...")
                    trainer = ModelTrainer(
                        dataframe=df,
                        yaml_config_path=self.train_cfg.get("config_path"),
                        target_column=self.train_cfg.get("target_column"),
                    )
                    models = trainer.train_all_models()
                    logger.info("✅ Models trained (fallback).")
                    return models
                else:
                    raise FileNotFoundError(f"No models found in {source_dir}")

            # ✅ 3. Load existing models
            logger.info(f"📁 Loading models from {source_dir}...")
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if not (file.endswith(".pkl") or file.endswith(".h5")):
                        continue

                    model_name = os.path.splitext(file)[0]
                    if not evaluate_all and model_name != selected_model:
                        continue

                    model_path = os.path.join(root, file)
                    try:
                        if file.endswith(".pkl"):
                            with open(model_path, "rb") as f:
                                model = pickle.load(f)
                        else:
                            model = load_model(model_path)
                        models[model_name] = model
                        logger.info(f"✅ Loaded model: {model_name} ({file})")
                    except Exception as load_err:
                        logger.warning(f"⚠️ Failed to load {file}: {load_err}")

            if not models:
                raise CustomException("No valid models loaded for evaluation.")

            return models

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise CustomException(e)

    def run(self):
        """Execute full evaluation pipeline."""
        try:
            logger.info("🚀 Starting Evaluation Pipeline...")

            # Load cleaned data
            df = self._load_or_prepare_data()

            # Load models from artifacts or train if needed
            models = self._load_models(df=df)

            # Split test data same as training
            trainer = ModelTrainer(
                dataframe=df,
                yaml_config_path=self.train_cfg.get("config_path"),
                target_column=self.train_cfg.get("target_column"),
            )
            trainer._prepare_data(model_type="lstm")  # prepares X_test, y_test
            X_test, y_test = trainer.X_test, trainer.y_test

            # Evaluate each model
            evaluator = ModelEvaluator()
            results = {}

            for name, model in models.items():
                logger.info(f"📊 Evaluating model: {name}")
                if name == "logistic_regression":
                    trainer._prepare_data(model_type="logistic_regression")
                    X_test, y_test = trainer.X_test, trainer.y_test
                else:
                    trainer._prepare_data(model_type="lstm")
                    X_test, y_test = trainer.X_test, trainer.y_test

                metrics = evaluator.evaluate(model, X_test, y_test, name)
                results[name] = metrics

            # Determine best model
            best_model_name, best_metrics = evaluator.get_best_model(results)
            best_model_obj = models[best_model_name]

            # Save metrics and best model
            evaluator.save_results(results, output_dir=self.eval_cfg.get("save_dir"))
            evaluator.save_best_model(
                best_model_name,
                best_model_obj,
                output_dir=self.eval_cfg.get("best_model_dir"),
            )

            logger.info(f"🏁 Evaluation complete. Best model: {best_model_name}")
            return results, best_model_name, best_metrics

        except Exception as e:
            logger.error(f"❌ Evaluation pipeline failed: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        pipeline = EvaluationPipeline(
            pipeline_config_path="configs/pipeline_params.yaml"
        )
        pipeline.run()
    except Exception as e:
        logger.error(f"🔥 Fatal error in EvaluationPipeline: {e}")
