import os
import torch  # 确保正确导入 torch
import pandas as pd
import chardet
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tabulate import tabulate

def load_data(file_path, sample_frac=0.2):
    # 检测文件编码
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']
    confidence = detected['confidence']
    print(f"Detected file encoding: {encoding} (confidence: {confidence})")

    # 如果编码检测不可靠，尝试其他编码
    if encoding is None or confidence < 0.8:
        encodings = ['utf-8', 'latin1', 'cp1252', 'gbk']
        for enc in encodings:
            try:
                data = pd.read_csv(file_path, encoding=enc)
                print(f"Successfully read file with encoding: {enc}")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            raise UnicodeDecodeError("Unable to decode file with common encodings.")
    else:
        data = pd.read_csv(file_path, encoding=encoding)

    if data.isnull().any().any():
        data.dropna(inplace=True)
    return data.sample(frac=sample_frac, random_state=42)

class CachedSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64, cache_dir=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def plot_metrics(metrics, output_dir, epoch):
    plt.figure(figsize=(10, 5))
    epochs = len(metrics['train_loss'])
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), metrics['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), metrics['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metrics_epoch_{epoch}.png"))
    plt.show()

class EarlyStopping:
    def __init__(self, patience=2, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), path)
            self.best_epoch = epoch
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), path)
            self.counter = 0
            self.best_epoch = epoch

def train_epoch(model, data_loader, optimizer, device, scheduler, scaler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    total_loss, correct = 0, 0
    for batch in data_loader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch['labels'].to(device)).sum().item()
    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

def evaluate_model(model, data_loader, device, label_encoder):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='macro')
    recall = recall_score(true_labels, preds, average='macro')
    f1 = f1_score(true_labels, preds, average='macro')
    report = classification_report(true_labels, preds, target_names=label_encoder)
    return accuracy, precision, recall, f1, report

def print_evaluation_table(metrics, output_path=None):
    headers = ["Metric", "Value"]
    table = [
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["F1 Score", f"{metrics['f1']:.4f}"]
    ]
    print("\nModel Evaluation Results:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    if output_path:
        with open(output_path, "w") as f:
            f.write(tabulate(table, headers=headers, tablefmt="grid"))
