import os
import pytest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
import requests
from src.utils.utils import download_file, datasplit, sequence_split
from src.utils.exception import CustomException

@patch('requests.get')
@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
def test_download_file_success(mock_exists, mock_makedirs, mock_get):
    # Mocking a successful response
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b'test data']

    url = 'http://example.com/test.csv'
    dest_folder = 'test_folder'

    with patch('builtins.open', mock_open()) as mock_file:
        file_path = download_file(url, dest_folder)
        assert file_path == os.path.join(dest_folder, 'test.csv')
        mock_makedirs.assert_called_once_with(dest_folder)
        mock_get.assert_called_once_with(url, stream=True)
        mock_file.assert_called_once_with(os.path.join(dest_folder, 'test.csv'), 'wb')

@patch('requests.get')
def test_download_file_failure(mock_get):
    # Mocking a failed response
    mock_get.side_effect = requests.exceptions.RequestException("Test error")

    url = 'http://example.com/test.csv'
    dest_folder = 'test_folder'

    with pytest.raises(requests.exceptions.RequestException):
        download_file(url, dest_folder)

def test_datasplit():
    data = {'vector': [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([7, 8]), np.array([9, 10])],
            'sentiment': ['pos', 'neg', 'pos', 'neg', 'pos']}
    df = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = datasplit(df, test_size=0.2, random_state=42)

    assert X_train.shape == (4, 2)
    assert X_test.shape == (1, 2)
    assert y_train.shape == (4,)
    assert y_test.shape == (1,)

def test_datasplit_missing_columns():
    data = {'vector': [np.array([1, 2]), np.array([3, 4])]}
    df = pd.DataFrame(data)

    with pytest.raises(CustomException):
        datasplit(df)

@patch('pickle.dump')
@patch('os.makedirs')
def test_sequence_split(mock_makedirs, mock_pickle_dump):
    data = {'clean_text': ['this is a test', 'this is another test', 'one more test', 'final test case', 'the last one'],
            'sentiment': ['pos', 'neg', 'pos', 'neg', 'pos']}
    df = pd.DataFrame(data)
    config = {'clean_data': {'test_size': 0.2, 'random_state': 42},
              'text_preprocessing': {'num_words': 100, 'max_len': 10}}

    X_train_seq, X_test_seq, y_train, y_test, tokenizer = sequence_split(df, config)

    assert X_train_seq.shape == (4, 10)
    assert X_test_seq.shape == (1, 10)
    assert y_train.shape == (4,)
    assert y_test.shape == (1,)
    mock_makedirs.assert_called_with('artifacts/models', exist_ok=True)
    assert mock_pickle_dump.called
