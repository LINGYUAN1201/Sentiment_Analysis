# Sentiment Analysis

This project is a sentiment analysis system built using pre-trained transformer models (e.g., DistilBERT). It provides functionalities for training, testing, and predicting sentiment in various text data formats.

---

## Features

### 1. Sentiment Analysis
- **Goal**: Classify text into predefined sentiment categories (e.g., "Positive", "Neutral", "Negative").
- **Key Features**:
  - Load pre-trained transformer models (like DistilBERT).
  - Support both real-time input and batch predictions.

### 2. Training and Testing (`train_and_test.py`)
- **Train a Model**:
  - Train the sentiment analysis model using a labeled dataset (e.g., `all_data.csv`).
  - Save the trained model and tokenizer files to the `trained_model/` directory.
- **Track Performance**:
  - Evaluate the model on validation and test datasets.
  - Generate logs and visualizations for training metrics (e.g., loss, accuracy).
- **Use Case**: Fine-tune the model for custom datasets or specific domains.

### 3. Sentiment Prediction (`predict_sentiment.py`)

#### Mode 1: Analyze Predefined Texts
- Use a set of predefined sample texts to test the model.
- **Use Case**: Quickly check the model's performance on known examples.

#### Mode 2: Analyze Texts from a File
- Load and analyze texts from a CSV file (e.g., `all_data.csv` or `input_texts.csv`).
- Automatically detect file encoding to avoid errors.
- **Use Case**: Batch processing of multiple texts.

#### Mode 3: Real-Time Input
- Enter texts interactively and get sentiment predictions.
- Type `exit` to end the session.
- **Use Case**: Manual testing or as an interactive tool.

---

## Generated Files

### 1. Model Files (`trained_model/`)
- `best_model.pt`: Trained PyTorch model weights.
- `model.safetensors`: Optional, model weights in `safetensors` format.
- Tokenizer configuration files (`config.json`, `tokenizer.json`, etc.).

### 2. Logs and Metrics (`logs/`)
- `training_metrics.csv`: Contains training and validation loss/accuracy for each epoch.
- `training_loss_plot.png`: Training loss visualization.
- `validation_confusion_matrix.png`: Confusion matrix for validation predictions.

### 3. Prediction Outputs (`results/`)
- `test_results.csv`: Predictions for texts in a batch file.
- `prediction_logs.txt`: Logs of predictions for real-time or batch inputs.

---

## Project Workflow

### Step 1: Train the Model
Run the `train_and_test.py` script to train the model:
```bash
pip install -r requirements.txt
```

```bash
python train_and_test.py
```

OR

```python
from train_and_test import train_and_find_optimal_lr

if __name__ == "__main__":
    train_and_find_optimal_lr(
        data_file="all_data.csv",  
        output_dir="results",  
        model_save_dir="trained_model", 
        max_epochs=10,  
        batch_size=32, 
        base_lr=3e-5
    )

```

### Step 2: Predict Sentiments
Use `predict_sentiment.py` for sentiment prediction. 

#### Mode 1: Analyze Predefined Texts
```python
from predict_sentiment import main

# Run Mode 1: Analyze predefined texts
main(mode=1, model_dir="trained_model")
```
#### Mode 2: Analyze Texts from a File
```python
from predict_sentiment import main

# Run Mode 2: Analyze texts from a CSV file
main(mode=2, model_dir="trained_model", file_path="all_data.csv")
```
#### Mode 3: Real-Time Input
```python
from predict_sentiment import main

# Run Mode 3: Real-time input
main(mode=3, model_dir="trained_model")
```

## Techniques and Methods Used

### 1. **Natural Language Processing (NLP) Techniques**
- **Pre-trained Transformer Models**:
  - Utilized DistilBERT, a lighter and faster variant of BERT, for sentiment classification.
  - Fine-tuned the pre-trained model on a custom dataset to adapt it to specific domains or categories.

- **Text Tokenization**:
  - Applied subword tokenization techniques to represent text input for transformer models.
  - Used padding and truncation to handle variable-length text sequences.

- **Sentiment Classification**:
  - The model outputs probabilities for predefined classes (e.g., "Positive", "Neutral", "Negative") and selects the most probable class.

### 2. **Machine Learning Workflow**
- **Fine-tuning Pre-trained Models**:
  - Adjusted weights of the pre-trained DistilBERT model using labeled data to improve its performance on the target task.

- **Dynamic Learning Rate Scheduling**:
  - Implemented learning rate warm-up and decay strategies for efficient training and to prevent overfitting.

- **Early Stopping**:
  - Monitored validation performance to stop training when no significant improvement was observed, saving computation time.

### 3. **Model Evaluation and Analysis**
- **Accuracy and Loss Tracking**:
  - Measured both training and validation accuracy and loss across epochs to monitor model performance.

- **Confusion Matrix**:
  - Visualized model predictions against true labels to evaluate classification performance and identify common errors.

- **Visualization**:
  - Plotted loss curves to understand the training process and detect potential overfitting or underfitting.

### 4. **Data Handling**
- **Data Splitting**:
  - Divided the dataset into training, validation, and test sets for robust evaluation.

- **Text Cleaning and Preparation**:
  - Removed unnecessary characters and ensured proper formatting of text for tokenization.

- **Batch Processing**:
  - Used data batching to handle large datasets efficiently during both training and inference.

### 5. **File Encoding Handling**
- **Automatic Encoding Detection**:
  - Implemented automatic file encoding detection to read diverse text files and avoid decoding errors.

### 6. **Real-time and Batch Inference**
- **Batch Prediction**:
  - Supported processing large datasets in a single batch for efficient analysis.

- **Interactive Real-time Input**:
  - Enabled users to input text dynamically and receive sentiment predictions instantly.

### 7. **Model Deployment Preparation**
- **Model Serialization**:
  - Saved model weights and configurations in both PyTorch and safetensors formats for flexible deployment.


## About training data

This dataset contains the sentiments for financial news headlines from the perspective of a retail investor. Further details about the dataset can be found in: Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2014): “Good debt or bad debt: Detecting semantic orientations in economic texts.” Journal of the American Society for Information Science and Technology.
