import os
import re
import yaml
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.downloader import load as api_load
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config

# Ensure required NLTK data is available
# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)
# nltk.download("wordnet", quiet=True)

class CleanData:
    """Cleans, preprocesses, and vectorizes text data."""

    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        try:
            self.config_path = config_path
            self.config = load_config(config_path)

            clean_cfg = self.config.get("clean_data", {})
            self.local_data_file = clean_cfg.get("local_data_file")
            self.save_dir = clean_cfg.get("save_dir")
            self.test_size = clean_cfg.get("test_size")
            self.random_state = clean_cfg.get("random_state")
            self.w_e_model_name = clean_cfg.get("word_embedding_model_name")
            self.w_e_model_save_path = clean_cfg.get("word_embedding_model_save_path")

            os.makedirs(self.save_dir, exist_ok=True)
            self.logger = logger

        except Exception as e:
            raise CustomException(e)

    def _clean_text(self, text):
        """Lowercase and remove noise from text."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _label_sentiment(self, rating):
        """Label reviews as positive or negative."""
        return "positive" if rating >= 3.5 else "negative"

    def _load_or_download_model(self):
        """Load local embedding model or download if not present."""
        try:
            if os.path.exists(self.w_e_model_save_path):
                self.logger.info(f"Loading word embedding model from: {self.w_e_model_save_path}")
                return  KeyedVectors.load_word2vec_format(self.w_e_model_save_path, binary=False, no_header=True)
            self.logger.info(f"Downloading word embedding model: {self.w_e_model_name}")
            model = api_load(self.w_e_model_name)
            os.makedirs(os.path.dirname(self.w_e_model_save_path), exist_ok=True)
            model.save(self.w_e_model_save_path)
            self.logger.info(f"Model saved to {self.w_e_model_save_path}")
            return model
        except Exception as e:
            raise CustomException(e)

    def _preprocess_tokens(self, text):
        """Tokenize, remove stopwords, and lemmatize text."""
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def _sentence_vector(self, tokens, model):
        """Convert tokens into averaged sentence vector."""
        valid_words = [w for w in tokens if w in model]
        if not valid_words:
            return np.zeros(model.vector_size)
        return np.mean(model[valid_words], axis=0)

    def clean_data(self, save: bool = True):
        """Main method to clean, tokenize, and vectorize data.

        Args:
            save (bool): If True, save cleaned dataframe to CSV. If False, return only the DataFrame.
        """
        try:
            self.logger.info("Loading raw dataset...")
            df = pd.read_csv(self.local_data_file)

            self.logger.info("Starting data cleaning and tokenization...")
            df["sentiment"] = df["rating"].apply(self._label_sentiment)
            df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            df["clean_text"] = df["full_text"].apply(self._clean_text)
            df["tokens"] = df["clean_text"].apply(self._preprocess_tokens)

            self.logger.info("Loading or downloading word embedding model...")
            w2v_model = self._load_or_download_model()

            self.logger.info("Generating sentence vectors...")
            df["vector"] = df["tokens"].apply(lambda x: self._sentence_vector(x, w2v_model))

            # ✅ Optional saving
            if save:
                output_file = os.path.join(self.save_dir, "cleaned_data.csv")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file, index=False)
                self.logger.info(f"💾 Cleaned data saved successfully at: {output_file}")
            else:
                self.logger.info("⚙️ Skipping save step (save=False). Returning DataFrame only.")

            return df

        except Exception as e:
            self.logger.error("❌ Error occurred during data cleaning.")
            raise CustomException(e)
    def process_input(
        self,
        title,
        text=None,
        use_tokenizer=False,
        batch_mode=False,
        separator="|||"
    ):
        """
        Clean and vectorize new input(s) for inference or evaluation.

        Args:
            title (str): Title or titles (batch input if batch_mode=True).
            text (str, optional): Text or texts corresponding to titles.
            use_tokenizer (bool): If True, use pre-trained tokenizer (for sequence models).
            batch_mode (bool): If True, treat inputs as batch separated by 'separator'.
            separator (str): String delimiter to split multiple inputs in batch mode.

        Returns:
            pd.DataFrame: Cleaned and vectorized data with consistent structure.
        """
        try:
            import pickle
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.preprocessing.text import Tokenizer

            # --- Handle batch input ---
            if batch_mode:
                title_list = [t.strip() for t in title.split(separator)]
                text_list = [t.strip() for t in text.split(separator)] if text else ["" for _ in title_list]
                df = pd.DataFrame({"title": title_list, "text": text_list})
                self.logger.info(f"🧾 Batch mode: received {len(df)} samples")
            else:
                df = pd.DataFrame({"title": [title], "text": [text or ""]})

            # --- Clean and tokenize ---
            df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            df["clean_text"] = df["full_text"].apply(self._clean_text)
            df["tokens"] = df["clean_text"].apply(self._preprocess_tokens)

            # --- Use tokenizer or embedding model ---
            if use_tokenizer:
                tok_path = self.config["clean_data"].get("tokenizer_path", "artifacts/models/tokenizer.pkl")
                if not os.path.exists(tok_path):
                    raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
                with open(tok_path, "rb") as f:
                    tokenizer = pickle.load(f)

                text_cfg = self.config.get("text_preprocessing", {})
                max_len = text_cfg.get("max_len", 200)

                sequences = tokenizer.texts_to_sequences(df["clean_text"])
                padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
                df["sequence"] = list(padded)
                self.logger.info(f"✅ Processed {len(df)} samples using tokenizer")

            else:
                w2v_model = self._load_or_download_model()
                df["vector"] = df["tokens"].apply(lambda x: self._sentence_vector(x, w2v_model))
                self.logger.info(f"✅ Processed {len(df)} samples using embedding vectors")

            return df

        except Exception as e:
            self.logger.error(f"❌ Failed to process input: {e}")
            raise CustomException(e)
