"""
Inference Pipeline Module for Customer Review Sentiment Analysis

This module provides production-ready inference capabilities for making predictions with trained models.
It provides functionality to:
    - Load trained models from disk in multiple formats:
        * .h5 format for Keras/TensorFlow deep learning models (LSTM, CNN, CNN-LSTM)
        * .pkl format for scikit-learn traditional ML models (Logistic Regression)
    - Process new review inputs for prediction:
        * Single sample inference - predict sentiment for one review
        * Batch inference - predict sentiment for multiple reviews in one call
    - Automatically select appropriate preprocessing based on model type:
        * Tokenizer-based sequence preprocessing for deep learning models
        * Word embedding-based vector preprocessing for traditional ML models
    - Clean and preprocess input text using the same pipeline as training:
        * Text cleaning (lowercasing, removing URLs, special characters)
        * Tokenization and lemmatization
        * Vectorization or sequence generation
    - Generate sentiment predictions (positive/negative) with confidence scores
    - Support configurable batch processing with custom separators
    - Optionally save prediction results to CSV files for analysis
    - Handle errors gracefully with comprehensive logging

The InferencePipeline class orchestrates the complete inference workflow by reading configuration
from YAML files, loading the appropriate trained model, preprocessing inputs using the same
transformations as during training, and generating predictions. The module is designed for
production deployment and supports both interactive single predictions and high-throughput
batch processing scenarios.
"""

import os, sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config
from src.data.clean_data import CleanData


class InferencePipeline:
    """
    Runs inference for trained sentiment models (.h5 or .pkl).
    """

    def __init__(self, config_path: str = str(Path("configs/inference_pipeline.yaml"))):
        try:
            logger.info("🔧 Initializing InferencePipeline...")
            self.config = load_config(config_path)
            infer_cfg = self.config.get("inference_pipeline", {})

            self.model_path = infer_cfg.get(
                "model_path", "artifacts/best_model/best.h5"
            )
            self.clean_config_path = infer_cfg.get(
                "clean_config_path", "configs/config.yaml"
            )
            self.batch_separator = infer_cfg.get("batch_separator", "|||")
            self.save_results = infer_cfg.get("save_results", False)
            self.output_path = infer_cfg.get(
                "output_path", "artifacts/inference/inference_results.csv"
            )

            self.model = self._load_model()
            logger.info(
                f"✅ InferencePipeline initialized successfully with model: {self.model_path}"
            )

        except Exception as e:
            logger.exception("❌ Failed to initialize inference pipeline")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _load_model(self):
        """Load model (.h5 for DL, .pkl for classical ML)."""
        try:
            if self.model_path.endswith(".h5"):
                logger.info(f"Loading TensorFlow model: {self.model_path}")
                model = load_model(self.model_path)
                self.model_type = "h5"
            elif self.model_path.endswith(".pkl"):
                logger.info(f"Loading pickle model: {self.model_path}")
                with open(self.model_path, "rb") as f:
                    model = pickle.load(f)
                self.model_type = "pkl"
            else:
                raise ValueError(f"Unsupported model type for file: {self.model_path}")
            return model
        except Exception as e:
            logger.exception("❌ Error loading model")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _predict(self, df: pd.DataFrame):
        """Generate predictions based on model type."""
        try:
            if self.model_type == "h5":
                X = np.stack(df["sequence"].values)  # ✅ stack into (N, 200)
                preds = self.model.predict(X)
                preds = (preds > 0.5).astype(int).flatten()
            else:
                X = np.vstack(df["vector"].values)
                preds = self.model.predict(X)

            df["predicted_label"] = np.where(preds == 1, "positive", "negative")
            return df
        except Exception as e:
            logger.exception("❌ Error during prediction")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def run(self, title, text=None, batch_mode=False):
        """
        Run inference on a single or batch input.

        Args:
            title (str): Title or batch titles string (use separator if batch).
            text (str): Review text or batch texts string (use separator if batch).
            batch_mode (bool): Enable batch mode if multiple samples are provided.
        """
        try:
            logger.info("🚀 Starting inference process...")

            # Decide tokenizer usage based on model type
            use_tokenizer = self.model_type == "h5"

            # Prepare input data
            cleaner = CleanData(config_path=self.clean_config_path)
            df = cleaner.process_input(
                title=title,
                text=text,
                use_tokenizer=use_tokenizer,
                batch_mode=batch_mode,
                separator=self.batch_separator,
            )

            # Generate predictions
            df = self._predict(df)

            # Optionally save results
            if self.save_results:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                df.to_csv(self.output_path, index=False)
                logger.info(f"💾 Inference results saved at: {self.output_path}")

            logger.info("✅ Inference completed successfully")
            return df

        except Exception as e:
            logger.exception("❌ Inference pipeline failed")
            raise CustomException(e)


if __name__ == "__main__":
    pipe = InferencePipeline(config_path="configs/pipeline_params.yaml")

    # 🔹 Single sample (beauty domain, negative example)
    result_df = pipe.run(
        title="Foundation oxidized terribly",
        text="This foundation turned orange after a few minutes of application. It looked fine at first, but quickly darkened and made my skin patchy.",
        batch_mode=False,
    )
    print("\nSingle Sample Result:")
    print(result_df)

    # 🔹 Batch mode (multiple beauty-domain negative samples)
    title = (
        "Mascara clumps everywhere|||"
        "Caused irritation and redness|||"
        "Too expensive for the quality|||"
        "Pump stopped working|||"
        "Overhyped product"
    )

    text = (
        "The formula is too thick and clumps on my lashes no matter how carefully I apply it.|||"
        "After two days of use, my skin became itchy and red. Had to stop immediately.|||"
        "The packaging looks fancy, but the product performs like a cheap drugstore brand.|||"
        "The pump broke after two uses, making it impossible to get any product out.|||"
        "Everyone raved about it, but I saw no difference in my skin after a month."
    )

    batch_df = pipe.run(title=title, text=text, batch_mode=True)
    print("\nBatch Sample Results:")
    print(batch_df)
