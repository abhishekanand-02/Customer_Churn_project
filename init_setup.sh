#!/bin/bash

echo [$(date)]: "START"

# Creating project folder structure
echo [$(date)]: "Creating directory structure"

# List of file paths to be created
list_of_files=(
    "src/__init__.py"
    "src/components/__init__.py"
    "src/components/data_ingestion.py"
    "src/components/data_transformation.py"
    "src/components/model_trainer.py"
    "src/components/model_evaluation.py"
    "src/pipelines/__init__.py"
    "src/pipelines/training_pipeline.py"
    "src/pipelines/prediction_pipeline.py"
    "src/logger.py"
    "src/exception.py"
    "src/utils/utils.py"
    "src/utils/__init__.py"
    "requirements.txt"
    "requirements_dev.txt"
    "experiment/experiments.ipynb"
    ".gitignore"               # Added .gitignore file
    ".github/workflows/main.yaml"  # Added GitHub Actions workflow file
)

# Creating directories and files
for filepath in "${list_of_files[@]}"; do
    filedir=$(dirname "$filepath")
    filename=$(basename "$filepath")

    # Create directory if it doesn't exist
    if [[ "$filedir" != "." ]]; then
        mkdir -p "$filedir"
    fi

    # Create an empty file if it doesn't exist or is empty
    if [[ ! -f "$filepath" || ! -s "$filepath" ]]; then
        touch "$filepath"
    fi
done

echo [$(date)]: "Project structure created successfully"
echo [$(date)]: "END"
