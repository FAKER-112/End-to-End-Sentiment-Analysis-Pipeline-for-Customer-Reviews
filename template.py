import os

# Define the directory structure relative to the current folder
structure = [
    "README.md",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    {"data": [{"raw":[]}, {"processed":[]}, {"external": []}]},
    {"notebooks": [
        "01_eda.ipynb",
        "02_feature_engineering.ipynb",
        "03_model_selection.ipynb"
    ]},
    {"src": [
        "__init__.py",
        {"data": [
            "__init__.py",
            "load_data.py",
            "clean_data.py",
            "feature_engineering.py"
        ]},
        {"models": [
            "__init__.py",
            "build_model.py",
            "train_model.py",
            "evaluate_model.py",
            "save_load.py"
        ]},
        {"utils": [
            "__init__.py",
            "config_parser.py",
            "logger.py",
            "metrics.py"
        ]},
        {"pipeline": [
            "__init__.py",
            "train_pipeline.py",
            "evaluate_pipeline.py",
            "predict_pipeline.py"
        ]}
    ]},
    {"configs": [
        "config.yaml",
        "model_params.yaml",
        "pipeline_params.yaml"
    ]},
    {"tests": [
        "test_data.py",
        "test_model.py",
        "test_pipeline.py"
    ]},
    {"scripts": [
        "run_train.py",
        "run_eval.py",
        "run_infer.py"
    ]},
    {"artifacts": [
        {"models" : []},
        {"metrics":[] },
        {"plots":[]}
    ]},
    {"docker": [
        "Dockerfile",
        "docker-compose.yaml"
    ]}
]


def create_structure(base_path, items):
    for item in items:
        if isinstance(item, str):
            file_path = os.path.join(base_path, item)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    pass
        elif isinstance(item, dict):
            for folder, contents in item.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, contents)


# Build structure inside the current directory
create_structure(os.getcwd(), structure)

print("✅ Project structure created successfully inside the current directory.")