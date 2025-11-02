import os
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, SpatialDropout1D,
    Conv1D, GlobalMaxPooling1D, MaxPooling1D
)
from sklearn.linear_model import LogisticRegression
from src.utils.logger import logger  # optional, if you want logging integration
from src.utils.exception import CustomException



class ModelBuilder:
    def __init__(self, config_path=None, config_dict=None):
        """Initialize ModelBuilder with YAML path or dict"""
        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

    def _build_embedding_layer(self, embedding_params):
        """
        Builds an Embedding layer dynamically from tokenizer or pre-trained matrix.
        """
        if embedding_params.get("embedding_matrix") is not None:
            return Embedding(
                input_dim=embedding_params["embedding_matrix"].shape[0],
                output_dim=embedding_params["embedding_matrix"].shape[1],
                weights=[embedding_params["embedding_matrix"]],
                input_length=embedding_params.get("max_len", 200),
                trainable=embedding_params.get("trainable", False)
            )
        else:
            return Embedding(
                input_dim=embedding_params.get("num_words", 10000),
                output_dim=embedding_params.get("embedding_dim", 100),
                input_length=embedding_params.get("max_len", 200),
                trainable=embedding_params.get("trainable", True)
            )

    def build_lstm_model(self, model_params=None):
        model_params = model_params or self.config.get('lstm_model', {})
        embedding_params = model_params.get('embedding', {})

        model = Sequential([
            self._build_embedding_layer(embedding_params),
            SpatialDropout1D(model_params.get('spatial_dropout', 0.3)),
            LSTM(
                model_params.get('lstm_units', 128),
                dropout=model_params.get('lstm_dropout', 0.2),
                recurrent_dropout=model_params.get('lstm_recurrent_dropout', 0.2)
            ),
            Dense(model_params.get('dense_units', 64), activation='relu'),
            Dropout(model_params.get('dropout_rate', 0.3)),
            Dense(1, activation='sigmoid')
        ])
        compile_params = model_params.get('compile', {})
        model.compile(
            loss=compile_params.get('loss', 'binary_crossentropy'),
            optimizer=compile_params.get('optimizer', 'adam'),
            metrics=compile_params.get('metrics', ['accuracy'])
        )
        return model

    def build_cnn_model(self, model_params=None):
        model_params = model_params or self.config.get('cnn_model', {})
        embedding_params = model_params.get('embedding', {})

        model = Sequential([
            self._build_embedding_layer(embedding_params),
            Conv1D(model_params.get('conv_filters', 128),
                   model_params.get('conv_kernel_size', 5), activation='relu'),
            GlobalMaxPooling1D(),
            Dense(model_params.get('dense_units', 64), activation='relu'),
            Dropout(model_params.get('dropout_rate', 0.3)),
            Dense(1, activation='sigmoid')
        ])
        compile_params = model_params.get('compile', {})
        model.compile(
            loss=compile_params.get('loss', 'binary_crossentropy'),
            optimizer=compile_params.get('optimizer', 'adam'),
            metrics=compile_params.get('metrics', ['accuracy'])
        )
        return model

    def build_cnn_lstm_model(self, model_params=None):
        model_params = model_params or self.config.get('cnn_lstm_model', {})
        embedding_params = model_params.get('embedding', {})

        model = Sequential([
            self._build_embedding_layer(embedding_params),
            Conv1D(model_params.get('conv_filters', 128),
                   model_params.get('conv_kernel_size', 5), activation='relu'),
            MaxPooling1D(pool_size=model_params.get('pool_size', 2)),
            LSTM(model_params.get('lstm_units', 64)),
            Dense(model_params.get('dense_units', 64), activation='relu'),
            Dropout(model_params.get('dropout_rate', 0.3)),
            Dense(1, activation='sigmoid')
        ])
        compile_params = model_params.get('compile', {})
        model.compile(
            loss=compile_params.get('loss', 'binary_crossentropy'),
            optimizer=compile_params.get('optimizer', 'adam'),
            metrics=compile_params.get('metrics', ['accuracy'])
        )
        return model

    def build_logistic_regression(self, model_params=None):
        model_params = model_params or self.config.get('logistic_regression', {})
        return LogisticRegression(
            max_iter=model_params.get('max_iter', 1000),
            class_weight=model_params.get('class_weight', 'balanced'),
            **model_params.get('additional_params', {})
        )

    def build_model(self, model_type, custom_params=None, show_summary=False):
        """Unified model builder interface"""
        model_builders = {
            'lstm': self.build_lstm_model,
            'cnn': self.build_cnn_model,
            'cnn_lstm': self.build_cnn_lstm_model,
            'logistic_regression': self.build_logistic_regression
        }

        if model_type not in model_builders:
            raise ValueError(f"Unknown model type: {model_type}. "
                             f"Available: {list(model_builders.keys())}")

        logger.info(f"Building {model_type.upper()} model...")
        model = model_builders[model_type](custom_params)
        if hasattr(model, 'summary') and show_summary:
            model.summary()
        return model
