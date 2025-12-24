# End-to-End NLP Project: Sentiment Analysis

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Training Pipeline](#training-pipeline)
- [Inference API](#inference-api)

## Overview

This project is an end-to-end Natural Language Processing (NLP) application that performs sentiment analysis on product reviews. It includes a complete MLOps pipeline for data ingestion, cleaning, model training, evaluation, and inference. The project is designed to be modular, configurable, and extensible.

The primary goal of this project is to demonstrate a production-ready approach to building and deploying machine learning models. It leverages best practices in software engineering and MLOps to create a robust and maintainable system.

**Key Features:**
- **Modular Pipeline:** The project is broken down into distinct stages: data loading, data cleaning, model training, and evaluation.
- **Configurable:** All aspects of the pipeline, from data sources to model hyperparameters, can be configured through YAML files.
- **Multiple Models:** The project supports training and evaluating multiple models, including Logistic Regression, LSTM, and CNN.
- **MLflow Integration:** The project is integrated with MLflow for experiment tracking and model management.
- **Inference API:** The project includes a simple API for making predictions on new data.

## Setup Instructions

To set up the project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/FAKER-112/End-to-End-Sentiment-Analysis-Pipeline-for-Customer-Reviews.git
cd End-to-End-Sentiment-Analysis-Pipeline-for-Customer-Reviews
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage the project's dependencies. You can create a virtual environment using `venv`:

```bash
python -m venv venv
source venv/bin/activate  
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Install NLTK Data

The project uses the NLTK library for text processing. You will need to download the required NLTK data. You can do this by running the following command in your terminal:

```bash
python -m nltk.downloader punkt stopwords wordnet
```

### 5. Download Word Embeddings

The project uses pre-trained word embeddings for the deep learning models. These will be downloaded automatically the first time you run the data cleaning pipeline.

## Training Pipeline

The training pipeline is responsible for orchestrating the entire process of training the sentiment analysis models. It is designed to be run from the command line and can be configured through YAML files.

### Running the Training Pipeline

To run the training pipeline, you can execute the following command from the root directory of the project:

```bash
python src/pipeline/train_pipeline.py
```

This will run the entire pipeline, from data ingestion to model evaluation, based on the configuration in `configs/pipeline_params.yaml`.

### Configuration

The training pipeline is configured through the `configs/pipeline_params.yaml` and `configs/config.yaml` files.

**`configs/config.yaml`:** This file contains the main configuration for the project, including data sources, model hyperparameters, and file paths.

- **`data_ingestion`:** You can specify the URL of the dataset to be downloaded in the `source_url` field.
- **`clean_data`:** This section contains parameters for the data cleaning process, such as the test set size and the name of the word embedding model.
- **`model_params`:** This section contains the hyperparameters for each of the models.

**`configs/pipeline_params.yaml`:** This file defines the stages of the training pipeline and the parameters for each stage.

- **`training_pipeline.pipeline.stages`:** This is a list of the stages to be executed, in order. The available stages are `load_data`, `clean_data`, `train_model`, and `evaluate_model`.
- **`training_pipeline.training`:** This section specifies the model to be trained and the path to the main configuration file.
- **`training_pipeline.evaluation`:** This section specifies the directories where the evaluation results and the best model should be saved.

### Pipeline Stages

The training pipeline consists of the following stages:

#### 1. `load_data`
This stage downloads the dataset from the URL specified in the configuration file, unzips it, and saves it as a CSV file.

#### 2. `clean_data`
This stage performs the following preprocessing steps on the raw data:
- **Text Cleaning:** Lowercasing, removing URLs, and removing non-alphabetic characters.
- **Sentiment Labeling:** Labeling reviews as "positive" or "negative" based on the rating.
- **Tokenization:** Splitting the text into individual words.
- **Stopword Removal:** Removing common English stopwords.
- **Lemmatization:** Reducing words to their base form.
- **Vectorization:** Converting the text into numerical vectors using pre-trained word embeddings.

#### 3. `train_model`
This stage trains the specified model on the preprocessed data. The project supports the following models:
- `logistic_regression`
- `lstm`
- `cnn`
- `cnn_lstm`

The model to be trained can be specified in the `configs/pipeline_params.yaml` file.

#### 4. `evaluate_model`
This stage evaluates the trained model on the test set and saves the evaluation metrics to a JSON file. The metrics calculated are:
- Accuracy
- Precision
- Recall
- F1-score

The best performing model is also saved to the `artifacts/best_model` directory.

### MLflow Integration

The training pipeline is integrated with MLflow for experiment tracking. When you run the training pipeline, the following information will be logged to MLflow:

- **Parameters:** The hyperparameters of the model.
- **Metrics:** The evaluation metrics of the model.
- **Artifacts:** The trained model itself.

To view the MLflow UI, you can run the following command from the root directory of the project:

```bash
mlflow ui
```

This will start the MLflow tracking server, which you can access in your web browser at `http://localhost:5000`.

## Inference API

The inference API allows you to use the trained sentiment analysis model to make predictions on new, unseen data. The API is implemented in the `src/pipeline/inference_pipeline.py` module.

### Running the Inference Pipeline

You can run the inference pipeline from the command line to see examples of single and batch predictions:

```bash
python src/pipeline/inference_pipeline.py
```

This will run the examples in the `if __name__ == "__main__":` block of the script, which demonstrate how to use the `InferencePipeline` class to make predictions.

### Using the `InferencePipeline`

To use the inference pipeline in your own code, you can import the `InferencePipeline` class and call its `run` method.

**Example: Single Prediction**

```python
from src.pipeline.inference_pipeline import InferencePipeline

# Initialize the pipeline
pipe = InferencePipeline(config_path="configs/pipeline_params.yaml")

# Make a single prediction
result_df = pipe.run(
    title="This is a great product!",
    text="I am very happy with my purchase.",
    batch_mode=False
)

print(result_df)
```

**Example: Batch Prediction**

```python
from src.pipeline.inference_pipeline import InferencePipeline

# Initialize the pipeline
pipe = InferencePipeline(config_path="configs/pipeline_params.yaml")

# Define the batch data
titles = (
    "This is a great product!|||"
    "This is a terrible product."
)

texts = (
    "I am very happy with my purchase.|||"
    "I am very unhappy with my purchase."
)

# Make a batch prediction
batch_df = pipe.run(
    title=titles,
    text=texts,
    batch_mode=True
)

print(batch_df)
```

### Configuration

The inference pipeline is configured through the `configs/inference_pipeline.yaml` file (or `configs/pipeline_params.yaml` if you are using the training pipeline's config).

- **`model_path`:** The path to the trained model to be used for inference. By default, this is set to the best model saved by the training pipeline.
- **`clean_config_path`:** The path to the main configuration file, which is used to configure the data cleaning process.
- **`batch_separator`:** The separator to be used when making batch predictions.

### Input and Output

**Input:**

The `run` method of the `InferencePipeline` class takes the following arguments:

- **`title` (str):** The title of the review(s). For batch predictions, the titles should be separated by the `batch_separator`.
- **`text` (str):** The text of the review(s). For batch predictions, the texts should be separated by the `batch_separator`.
- **`batch_mode` (bool):** Whether to make a single prediction or a batch prediction.

**Output:**

The `run` method returns a pandas DataFrame with the following columns:

- **`title`:** The original title of the review.
- **`text`:** The original text of the review.
- **`full_text`:** The concatenated title and text.
- **`clean_text`:** The preprocessed text.
- **`tokens`:** The tokenized text.
- **`vector` or `sequence`:** The numerical representation of the text.
- **`predicted_label`:** The predicted sentiment of the review ("positive" or "negative").
