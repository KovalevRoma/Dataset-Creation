# Dataset Creation for Code Generation Model Fine-tuning

This project involves creating a dataset and fine-tuning a code generation model for generating Kotlin code. The steps followed in this project are inspired by a coding task focused on adapting models to specific programming languages.

## Project Structure

- **main.py**: The main Python script that performs the following tasks:
  - Loads and processes a dataset of coding problems.
  - Transforms the dataset to Kotlin format.
  - Tokenizes and prepares the data for model training.
  - Fine-tunes a code generation model (`Salesforce/codegen-350M-mono`) on the synthetic Kotlin dataset.
- **.gitignore**: Lists files and folders that Git should ignore, such as virtual environment files, log files, and build artifacts.

## Steps Performed

### 1. Dataset Preparation

We used a subset of the **`jinaai/code_exercises`** dataset, which contains coding problems in Python. The dataset was transformed into a Kotlin-compatible format by modifying problem prompts and solution syntax.

### 2. Tokenization and Data Formatting

The data was tokenized using the `AutoTokenizer` from Hugging Face's `transformers` library, and formatted with padding and attention masks suitable for model training. Labels were created with careful alignment to the input tokens to avoid shape mismatches during training.

### 3. Model Selection and Fine-tuning

The `Salesforce/codegen-350M-mono` model was chosen due to its suitability for code generation tasks. The model was fine-tuned on the synthetic Kotlin dataset using the following parameters:
   - **Batch size**: 2
   - **Epochs**: 1 (for a quick experiment)
   - **Learning rate**: default for `Trainer`

Training was conducted with the `Trainer` API, which simplified the training loop and loss calculations.

### 4. Evaluation

After fine-tuning, the model was tested on a simple Kotlin function prompt to assess code generation quality.

## How to Run

### Prerequisites

- Python 3.7 or later
- A virtual environment (recommended)
- transformers
- datasets
- torch

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Dataset-Creation.git
   cd Dataset-Creation
	python3 -m venv myenv
	pip install transformers datasets evaluate pytorch
```
