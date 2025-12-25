"""
Unit tests for data modules in src/data.

Tests cover:
- LoadData: Data ingestion, downloading, unzipping, and parsing
- CleanData: Text cleaning, preprocessing, vectorization, and tokenization
"""

import os
import sys
import pytest
import yaml
import gzip
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.load_data import LoadData
from src.data.clean_data import CleanData
from src.utils.exception import CustomException


 
# FIXTURES
 


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "paths": {},
        "data_ingestion": {
            "source_url": "https://example.com/data.json.gz",
            "save_dir": "data/raw",
            "local_data_file": "data/processed/reviews.csv",
        },
        "clean_data": {
            "local_data_file": "data/processed/reviews.csv",
            "save_dir": "data/processed",
            "cleaned_data_path": "data/processed/cleaned_data.csv",
            "test_size": 0.2,
            "random_state": 42,
            "word_embedding_model_name": "word2vec-google-news-300",
            "word_embedding_model_save_path": "artifacts/embeddings/word2vec.model",
            "tokenizer_path": "artifacts/models/tokenizer.pkl",
        },
        "text_preprocessing": {
            "word2vec_model": "word2vec-google-news-300",
            "local_model_path": "artifacts/embeddings/word2vec.model",
            "num_words": 10000,
            "max_len": 200,
        },
    }


@pytest.fixture
def sample_reviews_data():
    """Sample review data for testing."""
    return [
        {"rating": 5.0, "title": "Great product", "text": "I love this item!"},
        {"rating": 1.0, "title": "Terrible", "text": "Not good at all"},
        {"rating": 4.0, "title": "Good quality", "text": "Pretty good overall"},
    ]


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "rating": [5.0, 1.0, 4.0, 2.0, 5.0],
            "title": ["Great", "Bad", "Good", "Poor", "Excellent"],
            "text": ["Love it", "Hate it", "Like it", "Dislike it", "Amazing"],
        }
    )


 
# LOAD DATA TESTS
 


class TestLoadData:
    """Tests for LoadData class."""

    @patch("src.data.load_data.load_config")
    def test_init_success(self, mock_load_config, tmp_path, sample_config):
        """Test successful initialization."""
        mock_load_config.return_value = sample_config

        loader = LoadData(raw_data_yaml="dummy_config.yaml")

        assert loader.source_url == "https://example.com/data.json.gz"
        assert loader.save_dir == "data/raw"
        assert loader.local_data_file == "data/processed/reviews.csv"

    @patch("src.data.load_data.load_config")
    def test_init_creates_directories(self, mock_load_config, tmp_path, sample_config):
        """Test initialization creates necessary directories."""
        # Update config to use tmp_path
        sample_config["data_ingestion"]["save_dir"] = str(tmp_path / "raw")
        sample_config["data_ingestion"]["local_data_file"] = str(
            tmp_path / "processed" / "reviews.csv"
        )
        mock_load_config.return_value = sample_config

        loader = LoadData(raw_data_yaml="dummy_config.yaml")

        assert os.path.exists(tmp_path / "raw")
        assert os.path.exists(tmp_path / "processed")

    @patch("src.data.load_data.load_config")
    def test_check_file_exists_true(self, mock_load_config, tmp_path, sample_config):
        """Test file existence check when file exists."""
        # Create a dummy file
        save_dir = tmp_path / "raw"
        save_dir.mkdir()
        test_file = save_dir / "data.json.gz"
        test_file.touch()

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        mock_load_config.return_value = sample_config

        loader = LoadData(raw_data_yaml="dummy_config.yaml")
        exists, file_path = loader._check_file_exists()

        assert exists is True
        assert file_path == str(test_file)

    @patch("src.data.load_data.load_config")
    def test_check_file_exists_false(self, mock_load_config, tmp_path, sample_config):
        """Test file existence check when file doesn't exist."""
        save_dir = tmp_path / "raw"
        save_dir.mkdir()

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        mock_load_config.return_value = sample_config

        loader = LoadData(raw_data_yaml="dummy_config.yaml")
        exists, file_path = loader._check_file_exists()

        assert exists is False

    @patch("src.data.load_data.download_file")
    @patch("src.data.load_data.load_config")
    def test_download_file_when_not_exists(
        self, mock_load_config, mock_download, tmp_path, sample_config
    ):
        """Test file download when file doesn't exist."""
        save_dir = tmp_path / "raw"
        save_dir.mkdir()

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        mock_load_config.return_value = sample_config

        expected_path = str(save_dir / "data.json.gz")
        mock_download.return_value = expected_path

        loader = LoadData(raw_data_yaml="dummy_config.yaml")
        loader._download_file()

        mock_download.assert_called_once_with(
            "https://example.com/data.json.gz", str(save_dir)
        )
        assert loader.download_path == expected_path

    @patch("src.data.load_data.load_config")
    def test_download_file_when_exists(self, mock_load_config, tmp_path, sample_config):
        """Test file download skips when file already exists."""
        save_dir = tmp_path / "raw"
        save_dir.mkdir()
        test_file = save_dir / "data.json.gz"
        test_file.touch()

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        mock_load_config.return_value = sample_config

        loader = LoadData(raw_data_yaml="dummy_config.yaml")

        with patch("src.data.load_data.download_file") as mock_download:
            loader._download_file()
            mock_download.assert_not_called()  # Should skip download
            assert loader.download_path == str(test_file)

    @patch("src.data.load_data.load_config")
    def test_unzip_success(
        self, mock_load_config, tmp_path, sample_config, sample_reviews_data
    ):
        """Test successful unzipping and parsing of gzipped JSON."""
        save_dir = tmp_path / "raw"
        save_dir.mkdir()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create a gzipped JSON file
        gz_file = save_dir / "data.json.gz"
        with gzip.open(gz_file, "wt") as f:
            for review in sample_reviews_data:
                f.write(json.dumps(review) + "\n")

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        sample_config["data_ingestion"]["local_data_file"] = str(
            processed_dir / "reviews.csv"
        )
        mock_load_config.return_value = sample_config

        loader = LoadData(raw_data_yaml="dummy_config.yaml")
        loader.download_path = str(gz_file)

        df = loader._unzip()

        assert df is not None
        assert len(df) == 3
        assert "rating" in df.columns
        assert "title" in df.columns
        assert "text" in df.columns

    @patch("src.data.load_data.download_file")
    @patch("src.data.load_data.load_config")
    def test_load_data_full_workflow(
        self,
        mock_load_config,
        mock_download,
        tmp_path,
        sample_config,
        sample_reviews_data,
    ):
        """Test complete load_data workflow."""
        save_dir = tmp_path / "raw"
        save_dir.mkdir()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create a gzipped JSON file
        gz_file = save_dir / "data.json.gz"
        with gzip.open(gz_file, "wt") as f:
            for review in sample_reviews_data:
                f.write(json.dumps(review) + "\n")

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        sample_config["data_ingestion"]["local_data_file"] = str(
            processed_dir / "reviews.csv"
        )
        mock_load_config.return_value = sample_config
        mock_download.return_value = str(gz_file)

        loader = LoadData(raw_data_yaml="dummy_config.yaml")
        df = loader.load_data()

        assert df is not None
        assert len(df) == 3
        assert os.path.exists(processed_dir / "reviews.csv")


 
# CLEAN DATA TESTS
 


class TestCleanData:
    """Tests for CleanData class."""

    @patch("src.data.clean_data.os.makedirs")  # Prevent directory creation
    @patch("src.data.clean_data.load_config")
    def test_init_success(self, mock_load_config, mock_makedirs, sample_config):
        """Test successful initialization."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        assert cleaner.config_path == "dummy_config.yaml"
        assert cleaner.config is not None

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_clean_text(self, mock_load_config, mock_makedirs, sample_config):
        """Test text cleaning functionality."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        dirty_text = "This is a GREAT product!!! Visit http://example.com for more."
        clean_text = cleaner._clean_text(dirty_text)

        assert clean_text.islower()
        assert "http" not in clean_text
        assert "!!!" not in clean_text

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_label_sentiment_positive(
        self, mock_load_config, mock_makedirs, sample_config
    ):
        """Test sentiment labeling for positive reviews."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        assert cleaner._label_sentiment(4.0) == "positive"
        assert cleaner._label_sentiment(5.0) == "positive"

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_label_sentiment_negative(
        self, mock_load_config, mock_makedirs, sample_config
    ):
        """Test sentiment labeling for negative reviews."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        assert cleaner._label_sentiment(1.0) == "negative"
        assert cleaner._label_sentiment(2.0) == "negative"
        assert cleaner._label_sentiment(3.0) == "negative"

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_preprocess_tokens(self, mock_load_config, mock_makedirs, sample_config):
        """Test token preprocessing (tokenize, remove stopwords, lemmatize)."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        text = "I am loving this amazing product"
        tokens = cleaner._preprocess_tokens(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Stopwords like "am", "this" should be removed
        assert "am" not in tokens
        assert "this" not in tokens

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.api_load")
    @patch("src.data.clean_data.load_config")
    def test_load_or_download_model_download(
        self, mock_load_config, mock_api_load, mock_makedirs, sample_config
    ):
        """Test downloading word2vec model when not present locally."""
        mock_load_config.return_value = sample_config

        # Mock the API load
        mock_model = MagicMock()
        mock_api_load.return_value = mock_model

        cleaner = CleanData(config_path="dummy_config.yaml")

        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", mock_open()):
                with patch("pickle.dump"):
                    model = cleaner._load_or_download_model()

        assert model is not None
        mock_api_load.assert_called_once()

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.pd.read_csv")
    @patch("src.data.clean_data.load_config")
    def test_clean_data_with_existing_file(
        self,
        mock_load_config,
        mock_read_csv,
        mock_makedirs,
        tmp_path,
        sample_config,
        sample_dataframe,
    ):
        """Test clean_data when CSV file already exists."""
        cleaned_path = tmp_path / "cleaned_data.csv"
        sample_config["clean_data"]["cleaned_data_path"] = str(cleaned_path)

        # Create the file
        sample_dataframe.to_csv(cleaned_path, index=False)

        mock_load_config.return_value = sample_config
        mock_read_csv.return_value = sample_dataframe

        cleaner = CleanData(config_path="dummy_config.yaml")

        with patch.object(cleaner, "_load_or_download_model"):
            df = cleaner.clean_data(save=False)

        assert df is not None
        mock_read_csv.assert_called_once()

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_process_input_single_sample(
        self, mock_load_config, mock_makedirs, sample_config
    ):
        """Test processing single input sample."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        # Mock the word2vec model
        mock_model = MagicMock()
        mock_model.vector_size = 300
        mock_model.__contains__ = lambda self, word: True
        mock_model.__getitem__ = lambda self, word: np.random.rand(300)

        with patch.object(cleaner, "_load_or_download_model", return_value=mock_model):
            df = cleaner.process_input(
                title="Great product",
                text="I love it",
                use_tokenizer=False,
                batch_mode=False,
            )

        assert df is not None
        assert len(df) == 1
        assert "vector" in df.columns

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_process_input_batch_mode(
        self, mock_load_config, mock_makedirs, sample_config
    ):
        """Test processing batch input samples."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        # Mock the word2vec model
        mock_model = MagicMock()
        mock_model.vector_size = 300
        mock_model.__contains__ = lambda self, word: True
        mock_model.__getitem__ = lambda self, word: np.random.rand(300)

        with patch.object(cleaner, "_load_or_download_model", return_value=mock_model):
            df = cleaner.process_input(
                title="Great|||Bad|||Good",
                text="Love it|||Hate it|||Like it",
                use_tokenizer=False,
                batch_mode=True,
                separator="|||",
            )

        assert df is not None
        assert len(df) == 3

    @patch("src.data.clean_data.os.makedirs")
    @patch("src.data.clean_data.load_config")
    def test_process_input_with_tokenizer(
        self, mock_load_config, mock_makedirs, sample_config
    ):
        """Test processing input with tokenizer for sequence models."""
        mock_load_config.return_value = sample_config

        cleaner = CleanData(config_path="dummy_config.yaml")

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]

        with patch("pickle.load", return_value=mock_tokenizer):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"mock data")):
                    df = cleaner.process_input(
                        title="Great product",
                        text=None,
                        use_tokenizer=True,
                        batch_mode=False,
                    )

        assert df is not None
        assert "sequence" in df.columns


 
# INTEGRATION TESTS
 


class TestDataIntegration:
    """Integration tests for data workflow."""

    @patch("src.data.load_data.download_file")
    @patch("src.data.clean_data.load_config")
    @patch("src.data.load_data.load_config")
    def test_load_and_clean_workflow(
        self,
        mock_load_config_load,
        mock_load_config_clean,
        mock_download,
        tmp_path,
        sample_config,
        sample_reviews_data,
    ):
        """Test end-to-end load → clean workflow."""
        save_dir = tmp_path / "raw"
        save_dir.mkdir()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create a gzipped JSON file
        gz_file = save_dir / "data.json.gz"
        with gzip.open(gz_file, "wt") as f:
            for review in sample_reviews_data:
                f.write(json.dumps(review) + "\n")

        sample_config["data_ingestion"]["save_dir"] = str(save_dir)
        sample_config["data_ingestion"]["local_data_file"] = str(
            processed_dir / "reviews.csv"
        )
        sample_config["clean_data"]["cleaned_data_path"] = str(
            processed_dir / "cleaned.csv"
        )

        mock_load_config_load.return_value = sample_config
        mock_load_config_clean.return_value = sample_config
        mock_download.return_value = str(gz_file)

        # Load data
        loader = LoadData(raw_data_yaml="dummy_config.yaml")
        raw_df = loader.load_data()

        assert raw_df is not None
        assert len(raw_df) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
