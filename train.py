import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和模型
from config.config import DATA_DIR, PROCESSED_DIR, MODEL_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, EMOTION_CLASSES
from models.fusion_model import AttentionFusion, SimpleFusion


class EmotionDataset(Dataset):
    def __init__(self, processed_dir, label_csv_path):
        self.processed_dir = processed_dir
        self.voice_dir = os.path.join(processed_dir, 'voice')
        self.face_dir = os.path.join(processed_dir, 'face')
        self.eda_dir = os.path.join(processed_dir, 'eda')

        # 加载标签
        self.label_df = pd.read_csv(label_csv_path)
        self.label_map = dict(zip(self.label_df['trial_id'], self.label_df['emotion']))  # 假设 'emotion' 是标签列，值为1-9

        # 获取所有文件路径
        self.voice_files = sorted([f for f in os.listdir(self.voice_dir) if f.endswith('.npy')])
        self.face_files = sorted([f for f in os.listdir(self.face_dir) if f.endswith('.npy')])
        self.eda_files = sorted([f for f in os.listdir(self.eda_dir) if f.endswith('.npy')])

        # 添加调试打印
        print("\n=== 调试信息 ===")
        print("前10个 voice 文件名:")
        for f in self.voice_files[:10]:
            print(f"  {f:40} → extract → {self.extract_trial_id(f)}")  # 注意：这里我把extract_trial_id作为类方法

        print("\n前10个 face 文件名:")
        for f in self.face_files[:10]:
            print(f"  {f:40} → extract → {self.extract_trial_id(f)}")

        print("\n前10个 eda 文件名:")
        for f in self.eda_files[:10]:
            print(f"  {f:40} → extract → {self.extract_trial_id(f)}")

        print("\nlabels.csv 里的前10个 trial_id:")
        print(list(self.label_map.keys())[:10])

        # 找到匹配的三模态样本
        self.common_files = self._find_common_files()
        print(f"找到 {len(self.common_files)} 个匹配的三模态样本")

        # 继续打印集合大小（在_find_common_files后移动过来或重复计算）
        voice_dict = {self.extract_trial_id(f): f for f in self.voice_files}
        face_dict = {self.extract_trial_id(f): f for f in self.face_files}
        eda_dict = {self.extract_trial_id(f): f for f in self.eda_files}
        print("\n提取到的 voice trial_ids 集合大小:", len(set(voice_dict.keys())))
        print("提取到的 face  trial_ids 集合大小:", len(set(face_dict.keys())))
        print("提取到的 eda   trial_ids 集合大小:", len(set(eda_dict.keys())))
        print("labels 中的 trial_id 数量:", len(self.label_map))

    def extract_trial_id(self, filename):  # 改为类方法，便于调用
        base = filename.replace('.npy', '')
        # 移除常见后缀如 '_C1', '_audio', '_emotion', '.wmv_audio' 等
        base = re.sub(r'(_C\d|_audio|_emotion|\.wmv_audio)', '', base)
        # 进一步清理，可能的模式
        match = re.search(r'Part_\d+_S_Trial\d+', base)
        return match.group(0) if match else base

    def _find_common_files(self):
        voice_dict = {self.extract_trial_id(f): f for f in self.voice_files}
        face_dict = {self.extract_trial_id(f): f for f in self.face_files}
        eda_dict = {self.extract_trial_id(f): f for f in self.eda_files}

        # 找到共同的 trial_id，且有标签
        common_trial_ids = set(voice_dict.keys()) & set(face_dict.keys()) & set(eda_dict.keys()) & set(
            self.label_map.keys())
        return sorted(list(common_trial_ids))

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        trial_id = self.common_files[idx]

        # 根据 trial_id 找到对应文件
        voice_file = os.path.join(self.voice_dir, [f for f in self.voice_files if trial_id in f][0])
        face_file = os.path.join(self.face_dir, [f for f in self.face_files if trial_id in f][0])
        eda_file = os.path.join(self.eda_dir, [f for f in self.eda_files if trial_id in f][0])

        # 加载三种模态的特征
        voice_features = np.load(voice_file)
        face_features = np.load(face_file)
        eda_features = np.load(eda_file)

        # 归一化特征
        voice_features = self._normalize(voice_features)
        face_features = self._normalize(face_features)
        eda_features = self._normalize(eda_features)

        # 加载真实标签（假设 emotion 从1到9，转换为0-based）
        emotion = self.label_map[trial_id]
        label = emotion - 1  # 假设 EMOTION_CLASSES 有9类，从0到8

        return {
            'voice': torch.tensor(voice_features, dtype=torch.float32),
            'face': torch.tensor(face_features, dtype=torch.float32),
            'eda': torch.tensor(eda_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _normalize(self, features):
        """归一化特征"""
        if np.std(features) > 0:
            return (features - np.mean(features)) / np.std(features)
        return features


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """训练模型"""
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_accuracy = 0.0
    best_model_path = os.path.join(MODEL_DIR, 'fusion', 'best_fusion_model.pth')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 训练阶段
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader):
            voice = batch['voice'].to(device)
            face = batch['face'].to(device)
            eda = batch['eda'].to(device)
            labels = batch['label'].to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            logits, _ = model(voice, face, eda)

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * voice.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                voice = batch['voice'].to(device)
                face = batch['face'].to(device)
                eda = batch['eda'].to(device)
                labels = batch['label'].to(device)

                # 前向传播
                logits, _ = model(voice, face, eda)

                # 计算损失
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * voice.size(0)

                # 计算预测
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # 计算准确率
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)

        # 计算其他指标
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型，准确率: {best_accuracy:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'fusion', 'training_curves.png'))
    plt.show()

    return model


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            voice = batch['voice'].to(device)
            face = batch['face'].to(device)
            eda = batch['eda'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits, _ = model(voice, face, eda)

            # 计算预测
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(ticks=np.arange(len(EMOTION_CLASSES)), labels=EMOTION_CLASSES, rotation=45)
    plt.yticks(ticks=np.arange(len(EMOTION_CLASSES)), labels=EMOTION_CLASSES)
    plt.colorbar()

    # 添加数值标签
    for i in range(len(EMOTION_CLASSES)):
        for j in range(len(EMOTION_CLASSES)):
            plt.text(j, i, conf_matrix[i, j], ha='center', va='center',
                     color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'fusion', 'confusion_matrix.png'))
    plt.show()

    return accuracy, precision, recall, f1


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 标签文件路径
    label_csv_path = os.path.join(PROCESSED_DIR, 'labels.csv')
    print(f"Label CSV path: {label_csv_path}")
    print(f"Processed directory: {PROCESSED_DIR}")

    # 检查文件是否存在
    if not os.path.exists(label_csv_path):
        print(f"Error: Label CSV file not found at {label_csv_path}")
        return

    # 检查processed目录是否存在
    if not os.path.exists(PROCESSED_DIR):
        print(f"Error: Processed directory not found at {PROCESSED_DIR}")
        return

    # 检查子目录是否存在
    voice_dir = os.path.join(PROCESSED_DIR, 'voice')
    face_dir = os.path.join(PROCESSED_DIR, 'face')
    eda_dir = os.path.join(PROCESSED_DIR, 'eda')

    print(f"Voice directory exists: {os.path.exists(voice_dir)}")
    print(f"Face directory exists: {os.path.exists(face_dir)}")
    print(f"EDA directory exists: {os.path.exists(eda_dir)}")

    # 检查文件数量
    if os.path.exists(voice_dir):
        voice_files = [f for f in os.listdir(voice_dir) if f.endswith('.npy')]
        print(f"Number of voice files: {len(voice_files)}")

    if os.path.exists(face_dir):
        face_files = [f for f in os.listdir(face_dir) if f.endswith('.npy')]
        print(f"Number of face files: {len(face_files)}")

    if os.path.exists(eda_dir):
        eda_files = [f for f in os.listdir(eda_dir) if f.endswith('.npy')]
        print(f"Number of EDA files: {len(eda_files)}")

    # 创建数据集
    print("Creating dataset...")
    dataset = EmotionDataset(PROCESSED_DIR, label_csv_path)

    if len(dataset) == 0:
        print("数据集为空，无法继续训练。请检查文件名匹配逻辑。")
        return

    # 划分训练集、验证集和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 确定特征维度
    sample = dataset[0]
    voice_dim = sample['voice'].shape[0]
    face_dim = sample['face'].shape[0]
    eda_dim = sample['eda'].shape[0]

    print(f"特征维度: voice={voice_dim}, face={face_dim}, eda={eda_dim}")

    # 初始化模型
    model = AttentionFusion((voice_dim, face_dim, eda_dim), hidden_dim=256)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, 'fusion', 'final_fusion_model.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"最终模型保存到: {final_model_path}")

    # 评估模型
    print("\n评估模型性能...")
    evaluate_model(trained_model, test_loader, device)


if __name__ == '__main__':
    main()