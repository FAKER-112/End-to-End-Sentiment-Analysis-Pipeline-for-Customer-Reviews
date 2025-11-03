import os, sys
import yaml
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.data.load_data import LoadData
from src.data.clean_data import CleanData
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.logger import logger
from src.utils.exception import CustomException


class TrainingPipeline:
    """End-to-end orchestrator for the ML training workflow."""

    def __init__(self, pipeline_config_path="config/pipeline_params.yaml"):
        try:
            logger.info("🔧 Initializing Training Pipeline...")
            if not os.path.exists(pipeline_config_path):
                raise FileNotFoundError(f"Pipeline config not found at: {pipeline_config_path}")

            with open(pipeline_config_path, "r") as f:
                self.pipeline_config = yaml.safe_load(f).get("training_pipeline", {})

            # Parse configuration sections
            self.pipeline_stages = self.pipeline_config.get("pipeline", {}).get("stages", [])
            self.training_cfg = self.pipeline_config.get("training", {})
            self.eval_cfg = self.pipeline_config.get("evaluation", {})

            logger.info(f"✅ Loaded pipeline configuration from {pipeline_config_path}")

        except Exception as e:
            logger.error("❌ Failed to initialize TrainingPipeline.")
            raise CustomException(e)

    # -------------------------------------------------------------------------
    def run(self):
        """Run the entire pipeline as per defined stages."""
        try:
            logger.info("🚀 Starting Training Pipeline Execution...")

            df = None
            trainer = None

            # Iterate over the configured stages
            for stage in self.pipeline_stages:
                logger.info(f"▶️ Running stage: {stage}")

                if stage == "load_data":
                    df = LoadData().load_data()

                elif stage == "clean_data":
                    df = CleanData().clean_data()

                elif stage == "train_model":
                    if df is None:
                        raise CustomException("No data available for training.")
                    trainer = ModelTrainer(
                        dataframe=df,
                        yaml_config_path=self.training_cfg.get("config_path"),
                        target_column=self.training_cfg.get("target_column"),
                    )
                    model_type = self.training_cfg.get("model_type")
                    model = trainer.train_model(model_type)
                    logger.info(f"✅ Model '{model_type}' trained successfully.")

                elif stage == "evaluate_model":
                    if trainer is None:
                        raise CustomException("Trainer not initialized before evaluation.")
                    evaluator = ModelEvaluator()
                    X_test, y_test = trainer.X_test, trainer.y_test
                    model_type = self.training_cfg.get("model_type")
                    metrics = evaluator.evaluate(trainer.models[model_type], X_test, y_test, model_type)
                    evaluator.save_results({model_type: metrics}, output_dir=self.eval_cfg.get("save_dir"))
                    evaluator.save_best_model(model_type, trainer.models[model_type], output_dir=self.eval_cfg.get("best_model_dir"))
                    logger.info(f"🏁 Evaluation completed for model: {model_type}")

                else:
                    logger.warning(f"⚠️ Unknown stage '{stage}' — skipping.")

            logger.info("🎉 Training pipeline completed successfully.")

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline(pipeline_config_path="configs/pipeline_params.yaml")
        pipeline.run()
    except Exception as e:
        logger.error(f"🔥 Fatal error in training pipeline: {e}")
