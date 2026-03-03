import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import EMOTION_CLASSES


class AttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=256):
        super(AttentionFusion, self).__init__()
        self.voice_dim, self.face_dim, self.eda_dim = input_dims

        # 模态特定投影层（统一到 hidden_dim）
        self.voice_proj = nn.Linear(self.voice_dim, hidden_dim)
        self.face_proj = nn.Linear(self.face_dim, hidden_dim)
        self.eda_proj = nn.Linear(self.eda_dim, hidden_dim)

        # 多头注意力（batch_first=True）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,  # 可以调大或调小
            batch_first=True,
            dropout=0.1
        )

        # 融合后全连接层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 分类头
        self.classifier = nn.Linear(hidden_dim, len(EMOTION_CLASSES))

        print(
            f"AttentionFusion 初始化完成：输入维度 {input_dims} → hidden {hidden_dim} → 输出 {len(EMOTION_CLASSES)} 类")

    def forward(self, voice, face, eda):
        # 投影到统一维度
        v = self.voice_proj(voice)  # [B, hidden]
        f = self.face_proj(face)
        e = self.eda_proj(eda)

        # 堆叠成序列 [B, 3, hidden]
        features = torch.stack([v, f, e], dim=1)

        # 自注意力
        attn_out, attn_weights = self.attention(features, features, features)

        # 展平 [B, 3*hidden]
        attn_flat = attn_out.reshape(attn_out.size(0), -1)

        # 融合处理
        fused = self.fusion_fc(attn_flat)

        # 分类
        logits = self.classifier(fused)

        return logits, fused  # 返回 logits 和融合特征（可选用于可视化）


class SimpleFusion(nn.Module):
    def __init__(self, input_dims):
        super(SimpleFusion, self).__init__()
        total_dim = sum(input_dims)
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(EMOTION_CLASSES))
        )
        print(f"SimpleFusion 初始化完成：输入总维度 {total_dim} → 输出 {len(EMOTION_CLASSES)} 类")

    def forward(self, voice, face, eda):
        combined = torch.cat([voice, face, eda], dim=1)
        logits = self.classifier(combined)
        return logits, combined


if __name__ == '__main__':
    # 测试代码
    voice_dim = 100
    face_dim = 1404  # 468*3
    eda_dim = 6  # 你的特征数，根据实际情况改

    model = AttentionFusion((voice_dim, face_dim, eda_dim), hidden_dim=256)

    # 模拟 batch=4
    v = torch.randn(4, voice_dim)
    f = torch.randn(4, face_dim)
    e = torch.randn(4, eda_dim)

    logits, fused = model(v, f, e)

    print(f"输入形状: voice={v.shape}, face={f.shape}, eda={e.shape}")
    print(f"输出形状: logits={logits.shape}, fused={fused.shape}")
    print(f"类别数: {len(EMOTION_CLASSES)}")