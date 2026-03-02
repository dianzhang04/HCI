import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import EMOTION_CLASSES

class AttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(AttentionFusion, self).__init__()
        # 三种模态的特征维度
        self.voice_dim, self.face_dim, self.eda_dim = input_dims
        
        # 模态特定的编码器
        self.voice_encoder = nn.Sequential(
            nn.Linear(self.voice_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.face_encoder = nn.Sequential(
            nn.Linear(self.face_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.eda_encoder = nn.Sequential(
            nn.Linear(self.eda_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 融合后处理
        self.fusion_processor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim // 2, len(EMOTION_CLASSES))
        
    def forward(self, voice_features, face_features, eda_features):
        # 编码各模态特征
        voice_encoded = self.voice_encoder(voice_features)
        face_encoded = self.face_encoder(face_features)
        eda_encoded = self.eda_encoder(eda_features)
        
        # 构建特征序列
        features = torch.stack([voice_encoded, face_encoded, eda_encoded], dim=1)
        
        # 应用注意力机制
        attn_output, _ = self.attention(features, features, features)
        
        # 展平注意力输出
        attn_output = attn_output.flatten(1)
        
        # 融合后处理
        fusion_output = self.fusion_processor(attn_output)
        
        # 分类
        logits = self.classifier(fusion_output)
        
        return logits, fusion_output

class SimpleFusion(nn.Module):
    """简单的拼接融合模型（备选）"""
    def __init__(self, input_dims):
        super(SimpleFusion, self).__init__()
        # 三种模态的特征维度
        self.voice_dim, self.face_dim, self.eda_dim = input_dims
        total_dim = self.voice_dim + self.face_dim + self.eda_dim
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(EMOTION_CLASSES))
        )
        
    def forward(self, voice_features, face_features, eda_features):
        # 拼接特征
        combined = torch.cat([voice_features, face_features, eda_features], dim=1)
        # 分类
        logits = self.classifier(combined)
        return logits, combined

if __name__ == '__main__':
    # 测试模型
    voice_dim = 100
    face_dim = 136  # 68个关键点，每个点2个坐标
    eda_dim = 14
    
    model = AttentionFusion((voice_dim, face_dim, eda_dim), 256)
    
    # 生成测试数据
    voice_input = torch.randn(32, voice_dim)
    face_input = torch.randn(32, face_dim)
    eda_input = torch.randn(32, eda_dim)
    
    # 前向传播
    logits, fusion_output = model(voice_input, face_input, eda_input)
    
    print(f"Input shapes: voice={voice_input.shape}, face={face_input.shape}, eda={eda_input.shape}")
    print(f"Output shapes: logits={logits.shape}, fusion_output={fusion_output.shape}")
    print(f"Number of emotion classes: {len(EMOTION_CLASSES)}")
