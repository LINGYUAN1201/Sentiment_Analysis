import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import load_data, CachedSentimentDataset, plot_metrics, EarlyStopping, eval_model, evaluate_model, print_evaluation_table

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_and_find_optimal_lr(data_file, output_dir="results", model_save_dir="trained_model", 
                              max_epochs=10, batch_size=32, base_lr=3e-5):
    """
    动态调整学习率并找到最佳训练轮次。
    """
    # 检查文件路径
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # 加载数据
    data = load_data(data_file, sample_frac=0.2)
    texts, labels = data['text'].tolist(), data['label'].factorize()[0]
    label_classes = data['label'].factorize()[1]  # 保存标签类别
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # 分割数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    train_dataset = CachedSentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = CachedSentimentDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(labels))).to(device)
    model.init_weights()

    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler(enabled=device.type == "cuda")
    early_stopping = EarlyStopping(patience=3, verbose=True)

    # 创建保存目录
    output_dir = f"{output_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    metrics = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_lr = base_lr
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        train_loss = 0
        model.train()
        with tqdm(train_loader, desc="Training", unit="batch") as t:
            for batch in t:
                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        metrics['train_loss'].append(train_loss)

        val_acc, val_loss = eval_model(model, test_loader, device)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        plot_metrics(metrics, output_dir, epoch + 1)
        early_stopping(val_loss, model, path=f"{model_save_dir}/best_model.pt", epoch=epoch + 1)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {early_stopping.best_epoch}")
            break

    model.load_state_dict(torch.load(f"{model_save_dir}/best_model.pt", weights_only=True))
    accuracy, precision, recall, f1, report = evaluate_model(model, test_loader, device, label_classes)
    evaluation_metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    print_evaluation_table(evaluation_metrics, output_path=f"{output_dir}/evaluation_results.txt")
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Model and tokenizer saved to {model_save_dir}")
    print(f"Best Learning Rate during training: {best_lr}")
