# Test Coverage Improvements

This document summarizes the improvements made to the test coverage of the repository.

## Files Modified

- `requirements.txt`: Added `pytest` and `coverage` to the project dependencies.
- `tests/test_utils.py`: Created a new test suite for the utility functions in `src/utils/utils.py`.

## New Tests and Covered Behaviors

The new test suite in `tests/test_utils.py` covers the following functions and behaviors:

### `download_file`
- **Successful Download**: Verifies that the function correctly downloads a file, creates the destination folder if it doesn't exist, and returns the correct file path.
- **Download Failure**: Ensures that the function raises a `requests.exceptions.RequestException` when the download fails.

### `datasplit`
- **Successful Data Splitting**: Checks that the function correctly splits a DataFrame into training and testing sets with the specified test size.
- **Missing Columns**: Verifies that the function raises a `CustomException` when the input DataFrame is missing required columns.

### `sequence_split`
- **Successful Sequence Splitting**: Confirms that the function correctly tokenizes and splits the data, and that the tokenizer is saved to the correct path.

## Coverage Results

The new tests have significantly improved the test coverage of the `src/utils` module. The coverage report shows the following:

- **`src/utils/utils.py`**: 89%
- **`src/utils` (module)**: 92%
