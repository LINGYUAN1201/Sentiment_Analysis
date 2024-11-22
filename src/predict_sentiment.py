import os
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from safetensors.torch import load_file  # 用于加载 safetensors 文件
import chardet  # 用于自动检测文件编码

# 设备配置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(model_dir):
    """
    加载模型和分词器
    """
    print("Loading tokenizer...")
    model_dir = model_dir.replace("\\", "/")  # 替换路径分隔符为 POSIX 风格
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    print("Tokenizer loaded.")

    print("Loading model...")
    model_path_pt = os.path.join(model_dir, "best_model.pt").replace("\\", "/")
    model_path_safetensors = os.path.join(model_dir, "model.safetensors").replace("\\", "/")

    if os.path.exists(model_path_safetensors):
        print("Using safetensors format.")
        state_dict = load_file(model_path_safetensors)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_dir, state_dict=state_dict
        ).to(device)
    elif os.path.exists(model_path_pt):
        print("Using PyTorch .pt format.")
        state_dict = torch.load(model_path_pt, map_location=device)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_dir, state_dict=state_dict
        ).to(device)
    else:
        raise FileNotFoundError("No valid model file found in the model directory.")

    print("Model loaded successfully.")
    return model, tokenizer

def predict_sentiment(text, model, tokenizer, label_classes):
    """
    对输入文本进行情感分类
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=-1).item()
    return label_classes[pred_label]

def batch_predict(texts, model, tokenizer, label_classes):
    """
    批量预测文本的情感类别
    """
    results = []
    for i, text in enumerate(texts):
        sentiment = predict_sentiment(text, model, tokenizer, label_classes)
        print(f"Processed {i+1}/{len(texts)}: {text} -> {sentiment}")
        results.append((text, sentiment))
    return results

def read_texts_from_file(file_path):
    """
    从 CSV 文件加载文本，自动检测编码
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    # 自动检测文件编码
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")

    # 使用检测到的编码读取文件
    data = pd.read_csv(file_path, encoding=encoding)
    if 'text' not in data.columns:
        raise ValueError("The file must have a 'text' column.")
    return data['text'].tolist()

def main(mode, model_dir="trained_model", file_path=None):
    """
    主函数，根据模式选择执行逻辑
    mode: 运行模式
        1: 预定义文本测试
        2: 从文件加载文本
        3: 实时输入文本
    model_dir: 模型保存路径
    file_path: 文本文件路径（仅在 mode=2 时需要）
    """
    # 定义标签类别（根据您的数据集调整）
    label_classes = ["Negative", "Neutral", "Positive"]

    # 加载模型和分词器
    model, tokenizer = load_model(model_dir)

    if mode == 1:
        # 模式 1：预定义文本测试
        texts = [
            "I love this product! It's amazing.",
            "This is the worst experience I've ever had.",
            "It's okay, but nothing special."
        ]
        predictions = batch_predict(texts, model, tokenizer, label_classes)
        for text, sentiment in predictions:
            print(f"Text: \"{text}\" -> Sentiment: {sentiment}")

    elif mode == 2:
        # 模式 2：从文件加载文本
        if not file_path:
            raise ValueError("File path must be provided for mode 2.")
        texts = read_texts_from_file(file_path)
        predictions = batch_predict(texts, model, tokenizer, label_classes)
        for text, sentiment in predictions:
            print(f"Text: \"{text}\" -> Sentiment: {sentiment}")

    elif mode == 3:
        # 模式 3：实时输入文本
        print("Enter texts to analyze sentiment (type 'exit' to quit):")
        while True:
            try:
                text = input("Enter a text: ").strip()
                if text.lower() == "exit":
                    print("Exiting real-time input mode.")
                    break
                sentiment = predict_sentiment(text, model, tokenizer, label_classes)
                print(f"Sentiment: {sentiment}")
            except KeyboardInterrupt:
                print("\nExiting real-time input mode.")
                break

    else:
        raise ValueError("Invalid mode. Choose 1, 2, or 3.")
