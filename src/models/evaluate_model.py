import os
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import save_model
from src.utils.logger import logger
from src.models.train_model import ModelTrainer
from src.data.load_data import LoadData
from src.data.clean_data import CleanData


class ModelEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate(self, model, X_test, y_test, model_name):
        """Compute evaluation metrics for a trained model"""
        y_pred = model.predict(X_test)
        if hasattr(y_pred, "ravel"):
            y_pred = (y_pred > 0.5).astype("int32")
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        self.metrics[model_name] = metrics
        logger.info(f"📊 {model_name} metrics: {metrics}")
        return metrics

    def get_best_model(self, results):
        """Select best model based on accuracy"""
        best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        best_metrics = results[best_model_name]
        logger.info(f"🏆 Best model: {best_model_name} (Accuracy={best_metrics['accuracy']:.4f})")
        return best_model_name, best_metrics

    def save_results(self, results, output_dir="artifacts/evaluation"):
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"💾 Evaluation results saved to {metrics_path}")

    def save_best_model(self, best_model_name, best_model_obj, output_dir="artifacts/best_model"):
        """Save the best model as .h5"""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "best.h5")

        try:
            # Save depending on model type
            if hasattr(best_model_obj, "save"):  # Keras or DL model
                best_model_obj.save(model_path)
            else:  # Scikit-learn model
                model_path = os.path.join(output_dir, f"best.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(best_model_obj, f)

            logger.info(f"💾 Best model saved to {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save best model: {e}")


if __name__ == "__main__":
    df = CleanData().clean_data()  # assuming LoadData() is already called inside CleanData
    trainer = ModelTrainer(df, "config.yaml", target_column="label")
    trained_models = trainer.train_all_models()

    evaluator = ModelEvaluator()
    results = {}

    for name, model in trained_models.items():
        X_test, y_test = trainer.X_test, trainer.y_test
        metrics = evaluator.evaluate(model, X_test, y_test, name)
        results[name] = metrics

    # Determine best model
    best_model_name, best_metrics = evaluator.get_best_model(results)
    best_model_obj = trained_models[best_model_name]

    # Save evaluation and model
    evaluator.save_results(results)
    evaluator.save_best_model(best_model_name, best_model_obj)
