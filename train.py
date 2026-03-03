import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import PROCESSED_DIR, MODEL_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, EMOTION_CLASSES
from models.fusion_model import AttentionFusion

# ==================== EmotionDataset ====================
class EmotionDataset(Dataset):
    def __init__(self, processed_dir, label_csv_path):
        self.processed_dir = processed_dir
        self.voice_dir = os.path.join(processed_dir, 'voice')
        self.face_dir = os.path.join(processed_dir, 'face')
        self.eda_dir = os.path.join(processed_dir, 'eda')

        print(f"加载标签文件: {label_csv_path}")
        if not os.path.exists(label_csv_path):
            raise FileNotFoundError(f"标签文件不存在: {label_csv_path}")

        self.label_df = pd.read_csv(label_csv_path)
        self.label_map = dict(zip(self.label_df['trial_id'].astype(str), self.label_df['emotion']))

        self.voice_files = sorted([f for f in os.listdir(self.voice_dir) if f.endswith('.npy')]) if os.path.exists(self.voice_dir) else []
        self.face_files = sorted([f for f in os.listdir(self.face_dir) if f.endswith('.npy')]) if os.path.exists(self.face_dir) else []
        self.eda_files = sorted([f for f in os.listdir(self.eda_dir) if f.endswith('.npy')]) if os.path.exists(self.eda_dir) else []

        print(f"\n文件统计: voice={len(self.voice_files)} face={len(self.face_files)} eda={len(self.eda_files)} labels={len(self.label_map)}")

        print("\n=== 前10个文件提取的 trial_id ===")
        for name, files in [("voice", self.voice_files), ("face", self.face_files), ("eda", self.eda_files)]:
            print(f"\n{name}:")
            for f in files[:10]:
                tid = self.extract_trial_id(f)
                print(f"  {f:50} → {tid}")

        print(f"\nlabels 前10个 trial_id: {list(self.label_map.keys())[:10]}")

        self.common_trial_ids = self._find_common_trial_ids()
        print(f"\n匹配样本数: {len(self.common_trial_ids)}")

        if len(self.common_trial_ids) == 0:
            print("警告：没有匹配样本！请检查文件名与 trial_id 是否一致")

    def extract_trial_id(self, filename):
        base = os.path.splitext(filename)[0]
        match = re.match(r'^(\d+)_', base)
        return match.group(1) if match else base

    def _find_common_trial_ids(self):
        voice_set = {self.extract_trial_id(f) for f in self.voice_files}
        face_set = {self.extract_trial_id(f) for f in self.face_files}
        eda_set = {self.extract_trial_id(f) for f in self.eda_files}
        label_set = set(self.label_map.keys())
        return sorted(list(voice_set & face_set & eda_set & label_set))

    def __len__(self):
        return len(self.common_trial_ids)

    def __getitem__(self, idx):
        trial_id = self.common_trial_ids[idx]

        voice_file = next(f for f in self.voice_files if trial_id == self.extract_trial_id(f))
        face_file = next(f for f in self.face_files if trial_id == self.extract_trial_id(f))
        eda_file = next(f for f in self.eda_files if trial_id == self.extract_trial_id(f))

        v = np.load(os.path.join(self.voice_dir, voice_file))
        f = np.load(os.path.join(self.face_dir, face_file))
        e = np.load(os.path.join(self.eda_dir, eda_file))

        v = (v - v.mean()) / (v.std() + 1e-8) if v.std() > 0 else v
        f = (f - f.mean()) / (f.std() + 1e-8) if f.std() > 0 else f
        e = (e - e.mean()) / (e.std() + 1e-8) if e.std() > 0 else e

        emotion = self.label_map.get(trial_id)
        if pd.isna(emotion):
            raise ValueError(f"trial_id {trial_id} 无有效标签")

        emotion_int = int(emotion)
        # 强制映射到 0~8
        if emotion_int in [1,2,3,4,5,6,7,8,9]:
            label = emotion_int - 1  # 正常映射
        elif emotion_int == 11:
            label = 7  # 11 → surprise (索引7)
            print(f"映射: {trial_id} emotion 11 → surprise (索引7)")
        elif emotion_int == 12:
            label = 8  # 12 → anxiety (索引8)
            print(f"映射: {trial_id} emotion 12 → anxiety (索引8)")
        else:
            print(f"警告: {trial_id} emotion {emotion_int} 异常，默认 neutral (索引3)")
            label = 3

        return {
            'voice': torch.tensor(v, dtype=torch.float32),
            'face': torch.tensor(f, dtype=torch.float32),
            'eda': torch.tensor(e, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    best_acc = 0.0
    patience = 10
    no_improve = 0
    best_path = os.path.join(MODEL_DIR, 'fusion', 'best_fusion_model.pth')
    final_path = os.path.join(MODEL_DIR, 'fusion', 'final_fusion_model.pth')

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            v = batch['voice'].to(device)
            f = batch['face'].to(device)
            e = batch['eda'].to(device)
            lbl = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(v, f, e)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * v.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        preds_all, lbls_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                v = batch['voice'].to(device)
                f = batch['face'].to(device)
                e = batch['eda'].to(device)
                lbl = batch['label'].to(device)

                logits, _ = model(v, f, e)
                loss = criterion(logits, lbl)
                val_loss += loss.item() * v.size(0)

                pred = logits.argmax(dim=1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)

                preds_all.extend(pred.cpu().numpy())
                lbls_all.extend(lbl.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  保存最佳模型: {best_path} (Val Acc: {best_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"早停触发：连续 {patience} epoch 未提升")
            break

    torch.save(model.state_dict(), final_path)
    print(f"训练结束，最终模型保存: {final_path}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.savefig(os.path.join(MODEL_DIR, 'fusion', 'training_curves.png'))
    plt.show()

    return model


# ==================== 评估函数 ====================
def evaluate_model(model, test_loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试集评估"):
            v = batch['voice'].to(device)
            f = batch['face'].to(device)
            e = batch['eda'].to(device)
            lbl = batch['label'].to(device)

            logits, _ = model(v, f, e)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(lbl.cpu().numpy())

    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(trues, preds)

    print(f"\n测试集结果:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(9), EMOTION_CLASSES, rotation=45)
    plt.yticks(range(9), EMOTION_CLASSES)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'fusion', 'confusion_matrix.png'))
    plt.show()


# ==================== 主函数 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    fusion_dir = os.path.join(MODEL_DIR, 'fusion')
    os.makedirs(fusion_dir, exist_ok=True)
    print(f"输出目录: {fusion_dir}")

    label_path = os.path.join(PROCESSED_DIR, 'labels_valid.csv')
    if not os.path.exists(label_path):
        label_path = os.path.join(PROCESSED_DIR, 'labels_all.csv')
        print("使用 labels_all.csv")

    dataset = EmotionDataset(PROCESSED_DIR, label_path)

    if len(dataset) == 0:
        print("无匹配样本，无法训练")
        return

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    sample = dataset[0]
    model = AttentionFusion(
        input_dims=(sample['voice'].shape[0], sample['face'].shape[0], sample['eda'].shape[0])
    ).to(device)

    # 计算权重（针对实际出现的标签）
    labels_list = []
    for i in range(len(dataset)):
        lbl = dataset[i]['label'].item() + 1  # 转回 1-based
        labels_list.append(lbl)

    unique_labels = np.unique(labels_list)
    print(f"实际出现的类别 (1-based): {sorted(unique_labels)}")

    # 生成 9 类权重（缺失类为 0）
    class_weights = torch.zeros(9, dtype=torch.float).to(device)
    if len(unique_labels) > 0:
        weights_np = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels_list
        )
        for idx, lbl in enumerate(unique_labels):
            class_weights[lbl - 1] = weights_np[idx]

    print(f"9 类权重: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

    print("\n开始测试集评估...")
    evaluate_model(trained_model, test_loader, device)


if __name__ == '__main__':
    main()