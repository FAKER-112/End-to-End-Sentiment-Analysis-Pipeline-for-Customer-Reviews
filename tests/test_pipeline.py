"""
Unit tests for pipeline modules in src/pipeline.

Tests cover:
- TrainingPipeline: End-to-end training orchestration
- EvaluationPipeline: Model evaluation and selection
- InferencePipeline: Production inference capabilities
"""

import os
import sys
import pytest
import yaml
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.evaluate_pipeline import EvaluationPipeline
from src.pipeline.inference_pipeline import InferencePipeline
from src.utils.exception import CustomException


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_training_config():
    """Sample training pipeline configuration."""
    return {
        "training_pipeline": {
            "pipeline": {
                "stages": ["load_data", "clean_data", "train_model", "evaluate_model"]
            },
            "training": {
                "config_path": "configs/config.yaml",
                "target_column": "sentiment",
                "model_type": "lstm",
            },
            "evaluation": {
                "save_dir": "artifacts/metrics",
                "best_model_dir": "artifacts/best_model",
            },
        }
    }


@pytest.fixture
def sample_evaluation_config():
    """Sample evaluation pipeline configuration."""
    return {
        "evaluation_pipeline": {
            "data": {"cleaned_data_path": "data/processed/cleaned_data.csv"},
            "models": {
                "source_dir": "artifacts/models",
                "evaluate_all": True,
                "selected_model": None,
                "train_models_if_missing": False,
                "train_all_models": False,
            },
            "evaluation": {
                "save_dir": "artifacts/metrics",
                "best_model_dir": "artifacts/best_model",
            },
            "training": {
                "config_path": "configs/config.yaml",
                "target_column": "sentiment",
            },
        }
    }


@pytest.fixture
def sample_inference_config():
    """Sample inference pipeline configuration."""
    return {
        "inference_pipeline": {
            "model_path": "artifacts/best_model/best.h5",
            "clean_config_path": "configs/config.yaml",
            "batch_separator": "|||",
            "save_results": False,
            "output_path": "artifacts/inference/results.csv",
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "title": ["Great product", "Terrible", "Amazing quality"],
            "text": ["I love this", "Not good", "Excellent item"],
            "sentiment": [1, 0, 1],
        }
    )


@pytest.fixture
def mock_model():
    """Mock model object."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.8], [0.2], [0.9]])
    return model


# ============================================================================
# TRAINING PIPELINE TESTS
# ============================================================================


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""

    def test_init_success(self, tmp_path, sample_training_config):
        """Test successful initialization with valid config."""
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        pipeline = TrainingPipeline(pipeline_config_path=str(config_file))

        assert pipeline.pipeline_config is not None
        assert (
            pipeline.pipeline_stages
            == sample_training_config["training_pipeline"]["pipeline"]["stages"]
        )
        assert pipeline.training_cfg["model_type"] == "lstm"
        assert pipeline.eval_cfg["save_dir"] == "artifacts/metrics"

    def test_init_missing_config_file(self):
        """Test initialization fails with missing config file."""
        with pytest.raises(CustomException):
            TrainingPipeline(pipeline_config_path="nonexistent_config.yaml")

    def test_init_invalid_yaml(self, tmp_path):
        """Test initialization fails with invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(CustomException):
            TrainingPipeline(pipeline_config_path=str(config_file))

    @patch("src.pipeline.train_pipeline.LoadData")
    @patch("src.pipeline.train_pipeline.CleanData")
    @patch("src.pipeline.train_pipeline.ModelTrainer")
    @patch("src.pipeline.train_pipeline.ModelEvaluator")
    def test_run_full_pipeline(
        self,
        mock_evaluator,
        mock_trainer,
        mock_clean,
        mock_load,
        tmp_path,
        sample_training_config,
        sample_dataframe,
        mock_model,
    ):
        """Test running complete training pipeline."""
        # Setup config
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        # Setup mocks
        mock_load_instance = Mock()
        mock_load_instance.load_data.return_value = sample_dataframe
        mock_load.return_value = mock_load_instance

        mock_clean_instance = Mock()
        mock_clean_instance.clean_data.return_value = sample_dataframe
        mock_clean.return_value = mock_clean_instance

        mock_trainer_instance = Mock()
        mock_trainer_instance.train_model.return_value = mock_model
        mock_trainer_instance.models = {"lstm": mock_model}
        mock_trainer_instance.X_test = np.array([[1, 2, 3]])
        mock_trainer_instance.y_test = np.array([1])
        mock_trainer.return_value = mock_trainer_instance

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate.return_value = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1": 0.93,
        }
        mock_evaluator.return_value = mock_evaluator_instance

        # Run pipeline
        pipeline = TrainingPipeline(pipeline_config_path=str(config_file))
        pipeline.run()

        # Verify all stages were called
        mock_load_instance.load_data.assert_called_once()
        mock_clean_instance.clean_data.assert_called_once()
        mock_trainer_instance.train_model.assert_called_once_with("lstm")
        mock_evaluator_instance.evaluate.assert_called_once()
        mock_evaluator_instance.save_results.assert_called_once()
        mock_evaluator_instance.save_best_model.assert_called_once()

    @patch("src.pipeline.train_pipeline.LoadData")
    def test_run_load_data_only(self, mock_load, tmp_path, sample_dataframe):
        """Test running pipeline with only load_data stage."""
        config = {
            "training_pipeline": {
                "pipeline": {"stages": ["load_data"]},
                "training": {},
                "evaluation": {},
            }
        }
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        mock_load_instance = Mock()
        mock_load_instance.load_data.return_value = sample_dataframe
        mock_load.return_value = mock_load_instance

        pipeline = TrainingPipeline(pipeline_config_path=str(config_file))
        pipeline.run()

        mock_load_instance.load_data.assert_called_once()

    @patch("src.pipeline.train_pipeline.ModelTrainer")
    def test_run_train_without_data_fails(self, mock_trainer, tmp_path):
        """Test training stage fails when no data is available."""
        config = {
            "training_pipeline": {
                "pipeline": {"stages": ["train_model"]},
                "training": {
                    "config_path": "configs/config.yaml",
                    "target_column": "sentiment",
                    "model_type": "lstm",
                },
                "evaluation": {},
            }
        }
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        pipeline = TrainingPipeline(pipeline_config_path=str(config_file))

        with pytest.raises(CustomException):
            pipeline.run()

    @patch("src.pipeline.train_pipeline.ModelEvaluator")
    def test_run_evaluate_without_trainer_fails(self, mock_evaluator, tmp_path):
        """Test evaluation stage fails when trainer is not initialized."""
        config = {
            "training_pipeline": {
                "pipeline": {"stages": ["evaluate_model"]},
                "training": {"model_type": "lstm"},
                "evaluation": {},
            }
        }
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        pipeline = TrainingPipeline(pipeline_config_path=str(config_file))

        with pytest.raises(CustomException):
            pipeline.run()

    def test_run_unknown_stage_skipped(self, tmp_path, sample_training_config):
        """Test unknown stages are skipped without errors."""
        sample_training_config["training_pipeline"]["pipeline"]["stages"] = [
            "unknown_stage"
        ]
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        pipeline = TrainingPipeline(pipeline_config_path=str(config_file))
        pipeline.run()  # Should complete without errors


# ============================================================================
# EVALUATION PIPELINE TESTS
# ============================================================================


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline class."""

    def test_init_success(self, tmp_path, sample_evaluation_config):
        """Test successful initialization with valid config."""
        config_file = tmp_path / "eval_pipeline.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))

        assert pipeline.pipeline_config is not None
        assert (
            pipeline.data_cfg["cleaned_data_path"] == "data/processed/cleaned_data.csv"
        )
        assert pipeline.model_cfg["evaluate_all"] is True
        assert pipeline.eval_cfg["save_dir"] == "artifacts/metrics"

    def test_init_missing_config_file(self):
        """Test initialization fails with missing config file."""
        with pytest.raises(CustomException):
            EvaluationPipeline(pipeline_config_path="nonexistent.yaml")

    def test_load_or_prepare_data_from_file(
        self, tmp_path, sample_evaluation_config, sample_dataframe
    ):
        """Test loading data from existing cleaned file."""
        config_file = tmp_path / "eval_pipeline.yaml"
        cleaned_path = tmp_path / "cleaned_data.csv"

        # Create dummy cleaned data file
        sample_dataframe.to_csv(cleaned_path, index=False)
        sample_evaluation_config["evaluation_pipeline"]["data"]["cleaned_data_path"] = (
            str(cleaned_path)
        )

        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))
        df = pipeline._load_or_prepare_data()

        assert df is not None
        assert len(df) == 3

    @patch("src.pipeline.evaluate_pipeline.CleanData")
    def test_load_or_prepare_data_clean_fallback(
        self, mock_clean, tmp_path, sample_evaluation_config, sample_dataframe
    ):
        """Test data cleaning fallback when file doesn't exist."""
        # Set nonexistent path
        sample_evaluation_config["evaluation_pipeline"]["data"][
            "cleaned_data_path"
        ] = "nonexistent.csv"
        config_file = tmp_path / "eval_pipeline.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        mock_clean_instance = Mock()
        mock_clean_instance.clean_data.return_value = sample_dataframe
        mock_clean.return_value = mock_clean_instance

        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))
        df = pipeline._load_or_prepare_data()

        assert df is not None
        mock_clean_instance.clean_data.assert_called_once()

    @patch("src.pipeline.evaluate_pipeline.ModelTrainer")
    def test_load_models_train_all_flag(
        self,
        mock_trainer,
        tmp_path,
        sample_evaluation_config,
        sample_dataframe,
        mock_model,
    ):
        """Test loading models with train_all_models flag set."""
        sample_evaluation_config["evaluation_pipeline"]["models"][
            "train_all_models"
        ] = True
        config_file = tmp_path / "eval_pipeline.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        mock_trainer_instance = Mock()
        mock_trainer_instance.train_all_models.return_value = {
            "lstm": mock_model,
            "cnn": mock_model,
        }
        mock_trainer.return_value = mock_trainer_instance

        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))
        models = pipeline._load_models(df=sample_dataframe)

        assert len(models) == 2
        mock_trainer_instance.train_all_models.assert_called_once()

    @patch("src.pipeline.evaluate_pipeline.pickle.load")
    @patch("src.pipeline.evaluate_pipeline.load_model")
    def test_load_models_from_disk(
        self,
        mock_load_h5,
        mock_load_pkl,
        tmp_path,
        sample_evaluation_config,
        mock_model,
    ):
        """Test loading models from disk."""
        config_file = tmp_path / "eval_pipeline.yaml"
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create actual model files
        (models_dir / "lstm.h5").touch()
        (models_dir / "logistic_regression.pkl").touch()

        sample_evaluation_config["evaluation_pipeline"]["models"]["source_dir"] = str(
            models_dir
        )
        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        mock_load_h5.return_value = mock_model

        # Use MagicMock for pickle.load which will be called when actual file is opened
        def mock_pickle_side_effect(*args, **kwargs):
            return mock_model

        mock_load_pkl.side_effect = mock_pickle_side_effect

        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))
        models = pipeline._load_models()

        assert len(models) == 2
        assert "lstm" in models
        assert "logistic_regression" in models

    def test_load_models_no_models_found_fails(
        self, tmp_path, sample_evaluation_config
    ):
        """Test error when no models are found and training is disabled."""
        config_file = tmp_path / "eval_pipeline.yaml"
        models_dir = tmp_path / "empty_models"
        models_dir.mkdir()

        sample_evaluation_config["evaluation_pipeline"]["models"]["source_dir"] = str(
            models_dir
        )
        sample_evaluation_config["evaluation_pipeline"]["models"][
            "train_models_if_missing"
        ] = False

        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))

        with pytest.raises(CustomException):
            pipeline._load_models()

    @patch("src.pipeline.evaluate_pipeline.ModelTrainer")
    @patch("src.pipeline.evaluate_pipeline.ModelEvaluator")
    def test_run_full_evaluation(
        self,
        mock_evaluator,
        mock_trainer,
        tmp_path,
        sample_evaluation_config,
        sample_dataframe,
        mock_model,
    ):
        """Test running complete evaluation pipeline."""
        config_file = tmp_path / "eval_pipeline.yaml"
        cleaned_path = tmp_path / "cleaned_data.csv"
        sample_dataframe.to_csv(cleaned_path, index=False)
        sample_evaluation_config["evaluation_pipeline"]["data"]["cleaned_data_path"] = (
            str(cleaned_path)
        )
        sample_evaluation_config["evaluation_pipeline"]["models"][
            "train_all_models"
        ] = True

        with open(config_file, "w") as f:
            yaml.dump(sample_evaluation_config, f)

        # Setup mocks
        mock_trainer_instance = Mock()
        mock_trainer_instance.train_all_models.return_value = {"lstm": mock_model}
        mock_trainer_instance.X_test = np.array([[1, 2, 3]])
        mock_trainer_instance.y_test = np.array([1])
        mock_trainer.return_value = mock_trainer_instance

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate.return_value = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1": 0.93,
        }
        mock_evaluator_instance.get_best_model.return_value = (
            "lstm",
            {"accuracy": 0.95},
        )
        mock_evaluator.return_value = mock_evaluator_instance

        # Run pipeline
        pipeline = EvaluationPipeline(pipeline_config_path=str(config_file))
        results, best_name, best_metrics = pipeline.run()

        # Verify
        assert best_name == "lstm"
        assert results is not None
        mock_evaluator_instance.evaluate.assert_called()
        mock_evaluator_instance.save_results.assert_called_once()
        mock_evaluator_instance.save_best_model.assert_called_once()


# ============================================================================
# INFERENCE PIPELINE TESTS
# ============================================================================


class TestInferencePipeline:
    """Tests for InferencePipeline class."""

    @patch("src.pipeline.inference_pipeline.load_model")
    @patch("src.pipeline.inference_pipeline.load_config")
    def test_init_success_h5_model(
        self,
        mock_load_config,
        mock_load_model,
        sample_inference_config,
        mock_model,
        tmp_path,
    ):
        """Test successful initialization with .h5 model."""
        mock_load_config.return_value = sample_inference_config
        mock_load_model.return_value = mock_model

        # Create dummy model file
        model_path = tmp_path / "best.h5"
        model_path.touch()
        sample_inference_config["inference_pipeline"]["model_path"] = str(model_path)

        pipeline = InferencePipeline(config_path="dummy_config.yaml")

        assert pipeline.model is not None
        assert pipeline.model_type == "h5"
        assert pipeline.batch_separator == "|||"

    @patch("src.pipeline.inference_pipeline.pickle.load")
    @patch("src.pipeline.inference_pipeline.load_config")
    def test_init_success_pkl_model(
        self,
        mock_load_config,
        mock_load_pkl,
        sample_inference_config,
        mock_model,
        tmp_path,
    ):
        """Test successful initialization with .pkl model."""
        sample_inference_config["inference_pipeline"]["model_path"] = "model.pkl"
        mock_load_config.return_value = sample_inference_config
        mock_load_pkl.return_value = mock_model

        with patch("builtins.open", mock_open(read_data=b"model_data")):
            pipeline = InferencePipeline(config_path="dummy_config.yaml")

        assert pipeline.model is not None
        assert pipeline.model_type == "pkl"

    @patch("src.pipeline.inference_pipeline.load_config")
    def test_init_unsupported_model_format(
        self, mock_load_config, sample_inference_config
    ):
        """Test initialization fails with unsupported model format."""
        sample_inference_config["inference_pipeline"]["model_path"] = "model.txt"
        mock_load_config.return_value = sample_inference_config

        with pytest.raises(CustomException):
            InferencePipeline(config_path="dummy_config.yaml")

    @patch("src.pipeline.inference_pipeline.load_config")
    @patch("src.pipeline.inference_pipeline.load_model")
    def test_predict_h5_model(
        self, mock_load_model, mock_load_config, sample_inference_config, tmp_path
    ):
        """Test prediction with h5 model."""
        mock_load_config.return_value = sample_inference_config

        # Create mock model with proper predict behavior
        mock_model_obj = Mock()
        mock_model_obj.predict.return_value = np.array([[0.8], [0.3], [0.6]])
        mock_load_model.return_value = mock_model_obj

        pipeline = InferencePipeline(config_path="dummy_config.yaml")

        # Create test dataframe with sequences
        df = pd.DataFrame(
            {
                "title": ["Good", "Bad", "Okay"],
                "sequence": [
                    np.array([1, 2, 3]),
                    np.array([4, 5, 6]),
                    np.array([7, 8, 9]),
                ],
            }
        )

        result = pipeline._predict(df)

        assert "predicted_label" in result.columns
        assert result["predicted_label"].iloc[0] == "positive"
        assert result["predicted_label"].iloc[1] == "negative"
        assert result["predicted_label"].iloc[2] == "positive"

    @patch("src.pipeline.inference_pipeline.load_config")
    @patch("src.pipeline.inference_pipeline.pickle.load")
    def test_predict_pkl_model(
        self, mock_load_pkl, mock_load_config, sample_inference_config
    ):
        """Test prediction with pkl model."""
        sample_inference_config["inference_pipeline"]["model_path"] = "model.pkl"
        mock_load_config.return_value = sample_inference_config

        # Create mock model
        mock_model_obj = Mock()
        mock_model_obj.predict.return_value = np.array([1, 0, 1])

        with patch("builtins.open", mock_open(read_data=b"model_data")):
            mock_load_pkl.return_value = mock_model_obj
            pipeline = InferencePipeline(config_path="dummy_config.yaml")

        # Create test dataframe with vectors
        df = pd.DataFrame(
            {
                "title": ["Good", "Bad", "Okay"],
                "vector": [
                    np.array([0.1, 0.2, 0.3]),
                    np.array([0.4, 0.5, 0.6]),
                    np.array([0.7, 0.8, 0.9]),
                ],
            }
        )

        result = pipeline._predict(df)

        assert "predicted_label" in result.columns
        assert result["predicted_label"].iloc[0] == "positive"
        assert result["predicted_label"].iloc[1] == "negative"

    @patch("src.pipeline.inference_pipeline.CleanData")
    @patch("src.pipeline.inference_pipeline.load_config")
    @patch("src.pipeline.inference_pipeline.load_model")
    def test_run_single_sample(
        self,
        mock_load_model,
        mock_load_config,
        mock_clean,
        sample_inference_config,
        mock_model,
    ):
        """Test running inference on single sample."""
        # Mock model to return single prediction
        mock_model_obj = Mock()
        mock_model_obj.predict.return_value = np.array([[0.8]])

        mock_load_config.return_value = sample_inference_config
        mock_load_model.return_value = mock_model_obj

        # Setup CleanData mock
        processed_df = pd.DataFrame(
            {
                "title": ["Great product"],
                "text": ["I love this"],
                "sequence": [np.array([1, 2, 3])],
            }
        )
        mock_clean_instance = Mock()
        mock_clean_instance.process_input.return_value = processed_df
        mock_clean.return_value = mock_clean_instance

        pipeline = InferencePipeline(config_path="dummy_config.yaml")
        result = pipeline.run(
            title="Great product", text="I love this", batch_mode=False
        )

        assert result is not None
        assert "predicted_label" in result.columns
        mock_clean_instance.process_input.assert_called_once()

    @patch("src.pipeline.inference_pipeline.CleanData")
    @patch("src.pipeline.inference_pipeline.load_config")
    @patch("src.pipeline.inference_pipeline.load_model")
    def test_run_batch_mode(
        self,
        mock_load_model,
        mock_load_config,
        mock_clean,
        sample_inference_config,
        mock_model,
    ):
        """Test running inference in batch mode."""
        mock_load_config.return_value = sample_inference_config
        mock_load_model.return_value = mock_model

        # Setup CleanData mock for batch
        processed_df = pd.DataFrame(
            {
                "title": ["Great product", "Bad item", "Okay"],
                "text": ["I love this", "Not good", "It's fine"],
                "sequence": [
                    np.array([1, 2, 3]),
                    np.array([4, 5, 6]),
                    np.array([7, 8, 9]),
                ],
            }
        )
        mock_clean_instance = Mock()
        mock_clean_instance.process_input.return_value = processed_df
        mock_clean.return_value = mock_clean_instance

        pipeline = InferencePipeline(config_path="dummy_config.yaml")
        result = pipeline.run(
            title="Great product|||Bad item|||Okay",
            text="I love this|||Not good|||It's fine",
            batch_mode=True,
        )

        assert result is not None
        assert len(result) == 3
        assert "predicted_label" in result.columns

    @patch("src.pipeline.inference_pipeline.CleanData")
    @patch("src.pipeline.inference_pipeline.load_config")
    @patch("src.pipeline.inference_pipeline.load_model")
    def test_run_save_results(
        self,
        mock_load_model,
        mock_load_config,
        mock_clean,
        sample_inference_config,
        tmp_path,
    ):
        """Test saving inference results to CSV."""
        output_path = tmp_path / "results.csv"
        sample_inference_config["inference_pipeline"]["save_results"] = True
        sample_inference_config["inference_pipeline"]["output_path"] = str(output_path)

        # Mock model to return single prediction
        mock_model_obj = Mock()
        mock_model_obj.predict.return_value = np.array([[0.8]])

        mock_load_config.return_value = sample_inference_config
        mock_load_model.return_value = mock_model_obj

        processed_df = pd.DataFrame(
            {"title": ["Great"], "text": ["Good"], "sequence": [np.array([1, 2, 3])]}
        )
        mock_clean_instance = Mock()
        mock_clean_instance.process_input.return_value = processed_df
        mock_clean.return_value = mock_clean_instance

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            pipeline = InferencePipeline(config_path="dummy_config.yaml")
            pipeline.run(title="Great", text="Good", batch_mode=False)

            mock_to_csv.assert_called_once()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline workflows."""

    @patch("src.pipeline.train_pipeline.LoadData")
    @patch("src.pipeline.train_pipeline.CleanData")
    @patch("src.pipeline.train_pipeline.ModelTrainer")
    @patch("src.pipeline.train_pipeline.ModelEvaluator")
    def test_training_to_evaluation_workflow(
        self,
        mock_eval,
        mock_trainer,
        mock_clean,
        mock_load,
        tmp_path,
        sample_training_config,
        sample_dataframe,
        mock_model,
    ):
        """Test end-to-end workflow from training to evaluation."""
        # Setup training pipeline
        config_file = tmp_path / "pipeline_params.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_training_config, f)

        # Setup mocks for training
        mock_load_instance = Mock()
        mock_load_instance.load_data.return_value = sample_dataframe
        mock_load.return_value = mock_load_instance

        mock_clean_instance = Mock()
        mock_clean_instance.clean_data.return_value = sample_dataframe
        mock_clean.return_value = mock_clean_instance

        mock_trainer_instance = Mock()
        mock_trainer_instance.train_model.return_value = mock_model
        mock_trainer_instance.models = {"lstm": mock_model}
        mock_trainer_instance.X_test = np.array([[1, 2, 3]])
        mock_trainer_instance.y_test = np.array([1])
        mock_trainer.return_value = mock_trainer_instance

        mock_eval_instance = Mock()
        mock_eval_instance.evaluate.return_value = {"accuracy": 0.95}
        mock_eval.return_value = mock_eval_instance

        # Run training pipeline
        train_pipeline = TrainingPipeline(pipeline_config_path=str(config_file))
        train_pipeline.run()

        # Verify training completed
        assert mock_trainer_instance.train_model.called
        assert mock_eval_instance.evaluate.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
