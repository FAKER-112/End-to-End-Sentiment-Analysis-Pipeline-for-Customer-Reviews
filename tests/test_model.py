"""
Unit tests for model modules in src/models.

Tests cover:
- ModelBuilder: Building various ML/DL models (LSTM, CNN, CNN-LSTM, Logistic Regression)
- ModelTrainer: Training models with data preparation and callbacks
- ModelEvaluator: Computing metrics, selecting best models, and saving results
"""

import os
import sys
import pytest
import yaml
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.build_model import ModelBuilder
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.exception import CustomException


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        "lstm_model": {
            "embedding": {
                "num_words": 10000,
                "embedding_dim": 100,
                "max_len": 200,
                "trainable": True,
            },
            "spatial_dropout": 0.3,
            "lstm_units": 128,
            "lstm_dropout": 0.2,
            "lstm_recurrent_dropout": 0.2,
            "dense_units": 64,
            "dropout_rate": 0.3,
            "compile": {
                "loss": "binary_crossentropy",
                "optimizer": "adam",
                "metrics": ["accuracy"],
            },
        },
        "cnn_model": {
            "embedding": {
                "num_words": 10000,
                "embedding_dim": 100,
                "max_len": 200,
                "trainable": True,
            },
            "conv_filters": 128,
            "conv_kernel_size": 5,
            "dense_units": 64,
            "dropout_rate": 0.3,
            "compile": {
                "loss": "binary_crossentropy",
                "optimizer": "adam",
                "metrics": ["accuracy"],
            },
        },
        "logistic_regression": {"max_iter": 1000, "class_weight": "balanced"},
    }


@pytest.fixture
def sample_training_config():
    """Sample training configuration."""
    return {
        "model_training": {
            "lstm": {"epochs": 5, "batch_size": 32, "validation_split": 0.2}
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "title": [
                "Great product",
                "Terrible",
                "Amazing quality",
                "Not good",
                "Love it",
            ],
            "text": [
                "I love this",
                "Not good at all",
                "Excellent item",
                "Bad quality",
                "So good",
            ],
            "sentiment": [1, 0, 1, 0, 1],
            "vector": [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, 0.6]),
                np.array([0.7, 0.8, 0.9]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.5, 0.6, 0.7]),
            ],
            "sequence": [
                np.array([1, 2, 3, 4, 5]),
                np.array([6, 7, 8, 9, 10]),
                np.array([11, 12, 13, 14, 15]),
                np.array([16, 17, 18, 19, 20]),
                np.array([21, 22, 23, 24, 25]),
            ],
        }
    )


# ============================================================================
# MODEL BUILDER TESTS
# ============================================================================


class TestModelBuilder:
    """Tests for ModelBuilder class."""

    def test_init_with_config_path(self, tmp_path, sample_model_config):
        """Test initialization with YAML config path."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_model_config, f)

        builder = ModelBuilder(config_path=str(config_file))

        assert builder.config is not None
        assert "lstm_model" in builder.config

    def test_init_with_config_dict(self, sample_model_config):
        """Test initialization with config dictionary."""
        builder = ModelBuilder(config_dict=sample_model_config)

        assert builder.config is not None
        assert builder.config == sample_model_config

    def test_init_missing_config_file(self):
        """Test initialization fails with missing config file."""
        with pytest.raises(FileNotFoundError):
            ModelBuilder(config_path="nonexistent_config.yaml")

    def test_init_no_config(self):
        """Test initialization fails when no config provided."""
        with pytest.raises(ValueError):
            ModelBuilder()

    def test_build_embedding_layer_with_matrix(self, sample_model_config):
        """Test building embedding layer with pre-trained matrix."""
        builder = ModelBuilder(config_dict=sample_model_config)

        embedding_matrix = np.random.rand(10000, 100)
        embedding_params = {
            "embedding_matrix": embedding_matrix,
            "max_len": 200,
            "trainable": False,
        }

        layer = builder._build_embedding_layer(embedding_params)

        assert layer is not None
        assert layer.input_dim == 10000
        assert layer.output_dim == 100

    def test_build_embedding_layer_without_matrix(self, sample_model_config):
        """Test building embedding layer from scratch."""
        builder = ModelBuilder(config_dict=sample_model_config)

        embedding_params = {
            "num_words": 5000,
            "embedding_dim": 50,
            "max_len": 150,
            "trainable": True,
        }

        layer = builder._build_embedding_layer(embedding_params)

        assert layer is not None
        assert layer.input_dim == 5000
        assert layer.output_dim == 50

    def test_build_lstm_model(self, sample_model_config):
        """Test building LSTM model."""
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_lstm_model()

        assert model is not None
        assert len(model.layers) > 0
        assert model.optimizer is not None

    def test_build_cnn_model(self, sample_model_config):
        """Test building CNN model."""
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_cnn_model()

        assert model is not None
        assert len(model.layers) > 0
        assert model.optimizer is not None

    def test_build_cnn_lstm_model(self, sample_model_config):
        """Test building CNN-LSTM hybrid model."""
        # Add CNN-LSTM config
        sample_model_config["cnn_lstm_model"] = {
            "embedding": {"num_words": 10000, "embedding_dim": 100, "max_len": 200},
            "conv_filters": 128,
            "conv_kernel_size": 5,
            "pool_size": 2,
            "lstm_units": 64,
            "dense_units": 64,
            "dropout_rate": 0.3,
            "compile": {
                "loss": "binary_crossentropy",
                "optimizer": "adam",
                "metrics": ["accuracy"],
            },
        }
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_cnn_lstm_model()

        assert model is not None
        assert len(model.layers) > 0

    def test_build_logistic_regression(self, sample_model_config):
        """Test building logistic regression model."""
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_logistic_regression()

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_model_unified_interface_lstm(self, sample_model_config):
        """Test unified build_model interface for LSTM."""
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_model("lstm")

        assert model is not None
        assert len(model.layers) > 0

    def test_build_model_unified_interface_cnn(self, sample_model_config):
        """Test unified build_model interface for CNN."""
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_model("cnn")

        assert model is not None

    def test_build_model_unified_interface_logistic(self, sample_model_config):
        """Test unified build_model interface for logistic regression."""
        builder = ModelBuilder(config_dict=sample_model_config)

        model = builder.build_model("logistic_regression")

        assert model is not None
        assert hasattr(model, "fit")

    def test_build_model_unknown_type(self, sample_model_config):
        """Test building model with unknown type raises error."""
        builder = ModelBuilder(config_dict=sample_model_config)

        with pytest.raises(ValueError):
            builder.build_model("unknown_model_type")

    def test_build_model_with_custom_params(self, sample_model_config):
        """Test building model with custom parameters."""
        builder = ModelBuilder(config_dict=sample_model_config)

        custom_params = {"max_iter": 500, "class_weight": None}

        model = builder.build_model("logistic_regression", custom_params=custom_params)

        assert model is not None
        assert model.max_iter == 500


# ============================================================================
# MODEL TRAINER TESTS
# ============================================================================


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    @patch("src.models.train_model.ModelBuilder")
    def test_init_success(
        self, mock_builder, tmp_path, sample_dataframe, sample_training_config
    ):
        """Test successful initialization."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )

        assert trainer.df is not None
        assert trainer.target_column == "sentiment"
        assert len(trainer.models) == 0
        assert len(trainer.histories) == 0

    def test_init_missing_config_file(self, sample_dataframe):
        """Test initialization fails with missing config file."""
        with pytest.raises(FileNotFoundError):
            ModelTrainer(
                dataframe=sample_dataframe,
                yaml_config_path="nonexistent.yaml",
                target_column="sentiment",
            )

    @patch("src.models.train_model.sequence_split")
    @patch("src.models.train_model.ModelBuilder")
    def test_prepare_data_for_lstm(
        self,
        mock_builder,
        mock_split,
        tmp_path,
        sample_dataframe,
        sample_training_config,
    ):
        """Test data preparation for LSTM model."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        # Mock sequence_split - returns 5 values including tokenizer
        mock_tokenizer = MagicMock()
        mock_split.return_value = (
            np.array([[1, 2, 3]]),  # X_train
            np.array([[4, 5, 6]]),  # X_test
            np.array([1]),  # y_train
            np.array([0]),  # y_test
            mock_tokenizer,  # tokenizer
        )

        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )

        trainer._prepare_data(model_type="lstm")

        assert trainer.X_train is not None
        assert trainer.X_test is not None
        mock_split.assert_called_once()

    @patch("src.models.train_model.datasplit")
    @patch("src.models.train_model.ModelBuilder")
    def test_prepare_data_for_logistic_regression(
        self,
        mock_builder,
        mock_split,
        tmp_path,
        sample_dataframe,
        sample_training_config,
    ):
        """Test data preparation for logistic regression."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        # Mock datasplit
        mock_split.return_value = (
            np.array([[0.1, 0.2]]),  # X_train
            np.array([[0.3, 0.4]]),  # X_test
            np.array([1]),  # y_train
            np.array([0]),  # y_test
        )

        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )

        trainer._prepare_data(model_type="logistic_regression")

        assert trainer.X_train is not None
        assert trainer.X_test is not None
        mock_split.assert_called_once()

    @patch("src.models.train_model.ModelBuilder")
    def test_setup_callbacks(self, mock_builder, tmp_path, sample_dataframe):
        """Test setting up training callbacks."""
        config = {
            "model_training": {
                "callbacks": {"early_stopping": {"monitor": "val_loss", "patience": 3}}
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )

        callbacks = trainer._setup_callbacks(config["model_training"])

        assert isinstance(callbacks, list)

    @patch("src.models.train_model.sequence_split")
    @patch("src.models.train_model.ModelBuilder")
    def test_train_model_lstm(
        self, mock_builder, mock_split, tmp_path, sample_dataframe
    ):
        """Test training LSTM model."""
        config = {
            "model_training": {
                "lstm": {"epochs": 2, "batch_size": 32, "validation_split": 0.2}
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Mock sequence_split - returns 5 values including tokenizer
        X_train = np.random.randint(0, 100, (10, 50))
        X_test = np.random.randint(0, 100, (3, 50))
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_test = np.array([1, 0, 1])
        mock_tokenizer = MagicMock()
        mock_split.return_value = (X_train, X_test, y_train, y_test, mock_tokenizer)

        # Mock ModelBuilder
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {"loss": [0.5, 0.4], "accuracy": [0.8, 0.85]}
        mock_model.fit.return_value = mock_history

        mock_builder_instance = MagicMock()
        mock_builder_instance.build_model.return_value = mock_model
        mock_builder.return_value = mock_builder_instance

        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )

        model = trainer.train_model("lstm")

        assert model is not None
        assert "lstm" in trainer.models
        mock_model.fit.assert_called_once()

    @patch("src.models.train_model.pickle.dump")  # Mock pickle.dump
    @patch("src.models.train_model.datasplit")
    @patch("src.models.train_model.ModelBuilder")
    def test_train_model_logistic_regression(
        self, mock_builder, mock_split, mock_pickle, tmp_path, sample_dataframe
    ):
        """Test training logistic regression model."""
        config = {"model_training": {}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Mock datasplit - returns 4 values (no tokenizer for logistic regression)
        X_train = np.random.rand(10, 3)
        X_test = np.random.rand(3, 3)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_test = np.array([1, 0, 1])
        mock_split.return_value = (X_train, X_test, y_train, y_test)

        # Mock ModelBuilder
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model

        mock_builder_instance = MagicMock()
        mock_builder_instance.build_model.return_value = mock_model
        mock_builder.return_value = mock_builder_instance

        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )

        model = trainer.train_model("logistic_regression")

        assert model is not None
        assert "logistic_regression" in trainer.models
        mock_model.fit.assert_called_once()
        mock_pickle.assert_called_once()  # Verify pickle.dump was called


# ============================================================================
# MODEL EVALUATOR TESTS
# ============================================================================


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_init(self):
        """Test initialization."""
        evaluator = ModelEvaluator()

        assert evaluator.metrics == {}

    def test_evaluate_keras_model(self):
        """Test evaluating a Keras model."""
        evaluator = ModelEvaluator()

        # Mock Keras model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.8], [0.3], [0.9], [0.2]])

        X_test = np.random.rand(4, 10)
        y_test = np.array([1, 0, 1, 0])

        metrics = evaluator.evaluate(mock_model, X_test, y_test, "test_model")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "test_model" in evaluator.metrics

    def test_evaluate_sklearn_model(self):
        """Test evaluating a scikit-learn model."""
        evaluator = ModelEvaluator()

        # Mock sklearn model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0])

        X_test = np.random.rand(4, 3)
        y_test = np.array([1, 0, 1, 0])

        metrics = evaluator.evaluate(mock_model, X_test, y_test, "lr_model")

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0  # Perfect prediction

    def test_get_best_model(self):
        """Test selecting best model based on accuracy."""
        evaluator = ModelEvaluator()

        results = {
            "model_a": {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.82,
                "f1": 0.82,
            },
            "model_b": {
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.91,
                "f1": 0.90,
            },
            "model_c": {
                "accuracy": 0.78,
                "precision": 0.75,
                "recall": 0.76,
                "f1": 0.75,
            },
        }

        best_name, best_metrics = evaluator.get_best_model(results)

        assert best_name == "model_b"
        assert best_metrics["accuracy"] == 0.92

    def test_save_results(self, tmp_path):
        """Test saving evaluation results to JSON."""
        evaluator = ModelEvaluator()
        output_dir = tmp_path / "evaluation"

        results = {
            "model_a": {"accuracy": 0.85, "precision": 0.83, "recall": 0.82, "f1": 0.82}
        }

        evaluator.save_results(results, output_dir=str(output_dir))

        metrics_file = output_dir / "metrics.json"
        assert metrics_file.exists()

        import json

        with open(metrics_file, "r") as f:
            saved_results = json.load(f)

        assert saved_results == results

    def test_save_best_model_keras(self, tmp_path):
        """Test saving best Keras model."""
        evaluator = ModelEvaluator()
        output_dir = tmp_path / "best_model"

        # Mock Keras model
        mock_model = MagicMock()
        mock_model.save = MagicMock()

        evaluator.save_best_model("lstm", mock_model, output_dir=str(output_dir))

        mock_model.save.assert_called_once()

    def test_save_best_model_sklearn(self, tmp_path):
        """Test saving best scikit-learn model."""
        evaluator = ModelEvaluator()
        output_dir = tmp_path / "best_model"

        # Mock sklearn model (no save method)
        mock_model = MagicMock(spec=[])  # Empty spec means no save method

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pickle.dump") as mock_pickle:
                evaluator.save_best_model("lr", mock_model, output_dir=str(output_dir))

                # Verify pickle.dump was called
                assert mock_pickle.called


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestModelIntegration:
    """Integration tests for model workflow."""

    @patch("src.models.train_model.sequence_split")
    @patch("src.models.train_model.ModelBuilder")
    def test_build_train_evaluate_workflow(
        self, mock_builder, mock_split, tmp_path, sample_dataframe
    ):
        """Test end-to-end workflow: build → train → evaluate."""
        config = {"model_training": {"lstm": {"epochs": 1, "batch_size": 32}}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Mock data split - returns 5 values including tokenizer
        X_train = np.random.randint(0, 100, (10, 50))
        X_test = np.random.randint(0, 100, (3, 50))
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_test = np.array([1, 0, 1])
        mock_tokenizer = MagicMock()
        mock_split.return_value = (X_train, X_test, y_train, y_test, mock_tokenizer)

        # Mock model
        mock_model = MagicMock()
        mock_model.fit.return_value = MagicMock(history={"loss": [0.5]})
        mock_model.predict.return_value = np.array([[0.8], [0.3], [0.9]])

        mock_builder_instance = MagicMock()
        mock_builder_instance.build_model.return_value = mock_model
        mock_builder.return_value = mock_builder_instance

        # Train
        trainer = ModelTrainer(
            dataframe=sample_dataframe,
            yaml_config_path=str(config_file),
            target_column="sentiment",
        )
        model = trainer.train_model("lstm")

        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, trainer.X_test, trainer.y_test, "lstm")

        assert model is not None
        assert "accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
