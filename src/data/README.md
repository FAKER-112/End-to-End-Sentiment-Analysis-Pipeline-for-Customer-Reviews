# Data Module

This module handles all data-related operations for the Customer Review Sentiment Analysis pipeline, including data ingestion, cleaning, preprocessing, and vectorization.

---

## 📁 Files Overview

### 1. [`load_data.py`](file:///c:/Users/User/Documents/Project/project_006/src/data/load_data.py)

**Purpose**: Handles raw data ingestion from remote sources.

**Key Features**:
- Downloads compressed review data from configured URLs
- Checks for existing files to avoid redundant downloads
- Unzips gzipped JSON Line (JSONL) files
- Extracts relevant fields (rating, title, text) from JSON records
- Saves a subset of data (first 1000 reviews) to CSV

**Main Class**: `LoadData`

**Configuration**: Reads from `configs/config.yaml`

**Usage**:
```python
from src.data.load_data import LoadData

# Initialize and load data
loader = LoadData(raw_data_yaml="configs/config.yaml")
df = loader.load_data()

# Returns: pandas DataFrame with columns [rating, title, text]
```

**Methods**:
- `__init__(raw_data_yaml)`: Initialize with config path
- `load_data()`: Execute end-to-end download and extraction
- `_download_file()`: Download file from source URL
- `_check_file_exists()`: Check if file already exists locally
- `_unzip()`: Extract and parse gzipped JSONL data

---

### 2. [`clean_data.py`](file:///c:/Users/User/Documents/Project/project_006/src/data/clean_data.py)

**Purpose**: Comprehensive text preprocessing and vectorization for sentiment analysis.

**Key Features**:
- Cleans raw text (lowercasing, removing URLs, special characters)
- Labels sentiment based on ratings (positive: ≥3.5, negative: <3.5)
- Tokenizes and lemmatizes text using NLTK
- Removes stopwords
- Generates sentence vectors using Word2Vec embeddings
- Supports both embedding-based and tokenizer-based preprocessing
- Handles both training data and inference inputs (single or batch)

**Main Class**: `CleanData`

**Configuration**: Reads from `configs/config.yaml`

**Usage**:

#### For Training Data:
```python
from src.data.clean_data import CleanData

# Initialize cleaner
cleaner = CleanData(config_path="configs/config.yaml")

# Clean and vectorize training data
df = cleaner.clean_data(save=True)
# Returns: DataFrame with [sentiment, full_text, clean_text, tokens, vector]
# Saves to: data/processed/cleaned_data.csv
```

#### For Inference (Single Sample):
```python
# Process new review for prediction
df = cleaner.process_input(
    title="Great product",
    text="I loved this item. Highly recommended!",
    use_tokenizer=False,  # Use Word2Vec embeddings
    batch_mode=False
)
```

#### For Inference (Batch Processing):
```python
# Process multiple reviews at once
titles = "Product A|||Product B|||Product C"
texts = "Review A text|||Review B text|||Review C text"

df = cleaner.process_input(
    title=titles,
    text=texts,
    use_tokenizer=False,
    batch_mode=True,
    separator="|||"
)
```

#### For Sequence Models (LSTM, CNN):
```python
# Use tokenizer for deep learning models
df = cleaner.process_input(
    title="Amazing quality",
    text="Best purchase ever!",
    use_tokenizer=True,  # Generate padded sequences
    batch_mode=False
)
# Returns: DataFrame with 'sequence' column (padded integer sequences)
```

**Methods**:
- `__init__(config_path)`: Initialize with config path
- `clean_data(save)`: Clean and vectorize training data
- `process_input(title, text, use_tokenizer, batch_mode, separator)`: Process new inputs
- `_clean_text(text)`: Remove noise and normalize text
- `_label_sentiment(rating)`: Convert rating to sentiment label
- `_preprocess_tokens(text)`: Tokenize, remove stopwords, lemmatize
- `_sentence_vector(tokens, model)`: Convert tokens to averaged vector
- `_load_or_download_model()`: Load/download Word2Vec model

---

## 🔄 Typical Workflow

### Training Pipeline:
```python
from src.data.load_data import LoadData
from src.data.clean_data import CleanData

# Step 1: Load raw data
loader = LoadData()
raw_df = loader.load_data()

# Step 2: Clean and vectorize
cleaner = CleanData()
clean_df = cleaner.clean_data(save=True)

# Output: Ready for model training
```

### Inference Pipeline:
```python
from src.data.clean_data import CleanData

# Initialize cleaner
cleaner = CleanData()

# Process new review
result = cleaner.process_input(
    title="Product title",
    text="Product review text",
    use_tokenizer=False,  # or True for DL models
    batch_mode=False
)

# Use result for prediction
```

---

## 📦 Dependencies

- **pandas**: DataFrame operations
- **numpy**: Numerical computations
- **nltk**: Tokenization, stopwords, lemmatization
- **gensim**: Word2Vec embedding models
- **tensorflow/keras**: Tokenizer and sequence preprocessing
- **yaml**: Configuration management

---

## ⚙️ Configuration

Both modules rely on `configs/config.yaml` for configuration:

```yaml
data_ingestion:
  source_url: "https://..."
  save_dir: "data/raw"
  local_data_file: "data/raw/reviews.csv"

clean_data:
  local_data_file: "data/raw/reviews.csv"
  save_dir: "data/processed"
  word_embedding_model_name: "glove-wiki-gigaword-100"
  word_embedding_model_save_path: "artifacts/word2vec/model.txt"
  tokenizer_path: "artifacts/models/tokenizer.pkl"
  
text_preprocessing:
  max_len: 200
```

---

## 📝 Output Files

### From `load_data.py`:
- `data/raw/<filename>.gz`: Downloaded compressed data
- `data/raw/reviews.csv`: Extracted review data

### From `clean_data.py`:
- `data/processed/cleaned_data.csv`: Cleaned and vectorized data
- `artifacts/word2vec/model.txt`: Cached Word2Vec model

---

## 🔍 Notes

1. **NLTK Data**: Ensure NLTK data is downloaded:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Word Embeddings**: The first run will download the Word2Vec model (can be slow). Subsequent runs will use the cached model.

3. **Memory Considerations**: Processing large datasets may require significant memory for vectorization.

4. **Sentiment Threshold**: Reviews with rating ≥ 3.5 are labeled "positive", others "negative". Adjust in `_label_sentiment()` if needed.

5. **Batch Processing**: Use batch mode for processing multiple samples efficiently in production.

---

## 🛠️ Troubleshooting

**Issue**: NLTK data not found  
**Solution**: Run `nltk.download('punkt')`, `nltk.download('stopwords')`, `nltk.download('wordnet')`

**Issue**: Word2Vec model download fails  
**Solution**: Check internet connection or manually download the model and place it in the configured path

**Issue**: Out of memory during vectorization  
**Solution**: Process data in smaller batches or reduce the dataset size

**Issue**: Tokenizer not found for sequence models  
**Solution**: Ensure the tokenizer is trained and saved during the training pipeline before inference

---

## 📚 Additional Resources

- [NLTK Documentation](https://www.nltk.org/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Pandas DataFrame Guide](https://pandas.pydata.org/docs/user_guide/dsintro.html)
