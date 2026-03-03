# 基于多模态情感识别与大模型共情回应的智能陪护系统

## 项目简介

本项目是一个基于多模态情感识别与大模型共情回应的智能陪护系统，集成了以下核心功能：

1. **多模态情感识别**：使用语音、面部表情和皮肤电信号（EDA/GSR）三种模态的数据进行情感分类
2. **深度学习模型**：采用注意力融合模型（AttentionFusion）融合不同模态的特征，提高情感识别准确性
3. **大语言模型集成**：基于识别到的情绪生成共情回应，提供情感支持
4. **交互式应用**：通过Streamlit构建用户友好的界面，实现实时情感识别和对话

项目旨在为用户提供一个智能、共情的陪护系统，能够理解用户的情感状态并给予适当的回应。

## 目录结构

```
HCI/
├── config/           # 配置文件
│   └── config.py     # 项目配置
├── models/           # 模型定义和训练结果
│   ├── fusion/       # 训练后的模型和结果
│   └── fusion_model.py  # 多模态融合模型
├── preprocessing/    # 数据预处理
│   ├── processed/    # 预处理后的数据
│   └── data_preprocessing.py  # 预处理脚本
├── app.py            # 应用程序主文件
├── emotion_prompts.py # 情感提示相关文件
├── llm_integration.py # 大语言模型集成
├── requirements.txt  # 依赖项文件
├── train.py          # 模型训练和评估脚本
└── README.md         # 项目说明文档
```

## 依赖项

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Librosa (语音处理)
- OpenCV (面部处理)
- MediaPipe (面部关键点检测)
- Scikit-learn (评估指标)
- Matplotlib (可视化)
- MNE (生理信号处理，可选)
- tqdm (进度条)

## 安装步骤

1. **克隆项目**

2. **创建虚拟环境**
   ```bash
   conda create -n HCI python=3.8
   conda activate HCI
   ```

3. **安装依赖**
   使用requirements.txt文件安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
   或者手动安装：
   ```bash
   pip install torch torchvision
   pip install numpy pandas librosa opencv-python mediapipe scikit-learn matplotlib tqdm
   # 可选：用于处理生理信号
   pip install mne
   ```

## 数据准备

### 数据集

本项目使用 MAHNOB-HCI 数据集，包含多模态情感数据。数据集应按照以下结构组织：

```
Sessions/
├── 2/
│   ├── session.xml        # 标签和元数据
│   ├── P1-Rec1-Audio_Section_2.wav  # 音频文件
│   ├── P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_2.avi  # 面部视频
│   └── Part_1_S_Trial1_emotion.bdf  # 生理信号
├── 4/
│   └── ...
└── ...
```

### 配置数据路径

在 `config/config.py` 中设置数据集路径：

```python
DATA_DIR = 'E:\\HCI-Tagging\\HCI-Tagging Databases\\Sessions'
```

## 数据预处理

运行数据预处理脚本，提取特征并生成标签：

```bash
cd preprocessing
python data_preprocessing.py
```

预处理步骤包括：

1. **标签提取**：从 session.xml 文件中提取情感标签（ valence, arousal, emotion 等）
2. **语音预处理**：提取 MFCC、Mel 频谱等特征
3. **面部预处理**：使用 MediaPipe 提取面部关键点
4. **EDA/GSR 预处理**：处理皮肤电信号，提取特征

预处理后的文件将保存在 `preprocessing/processed/` 目录中。

## 模型训练

运行训练脚本：

```bash
python train.py
```

训练过程包括：

1. **数据加载**：加载预处理后的多模态特征和标签
2. **模型初始化**：使用 AttentionFusion 模型融合多模态特征
3. **训练循环**：训练模型并保存最佳模型
4. **模型评估**：在测试集上评估模型性能
5. **结果可视化**：生成训练曲线和混淆矩阵

## 应用程序

### Streamlit应用

项目包含一个基于Streamlit的交互式应用，用于实时情感识别和共情对话：

```bash
streamlit run app.py
```

应用功能：

1. **多模态情感识别**：支持语音、面部表情和皮肤电信号的情感识别
2. **共情对话**：基于识别到的情绪生成共情回应
3. **实时交互**：通过聊天界面与用户进行实时对话
4. **系统状态监控**：显示设备状态、模型加载情况等信息

### 大语言模型集成

项目集成了大语言模型，用于生成共情回应：

- **llm_integration.py**：负责与大语言模型API的交互
- **emotion_prompts.py**：根据情绪状态生成相应的提示

### 配置大语言模型

在 `config/config.py` 中设置API密钥：

```python
API_KEY = 'your_api_key_here'
API_URL = 'https://api.openai.com/v1/chat/completions'
MODEL_NAME = 'gpt-3.5-turbo'
```

如果未配置API密钥，应用将使用模拟的LLM回应。

## 模型结构

本项目使用注意力融合模型（AttentionFusion），通过注意力机制自适应地融合不同模态的特征。模型结构如下：

1. **单模态特征提取**：对每个模态的特征进行编码
2. **注意力机制**：计算每个模态的注意力权重
3. **特征融合**：根据注意力权重融合多模态特征
4. **分类器**：使用融合后的特征进行情感分类

## 评估指标

模型评估使用以下指标：

- **准确率**：正确分类的样本比例
- **精确率**：预测为正类的样本中实际为正类的比例
- **召回率**：实际为正类的样本中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均值
- **混淆矩阵**：展示各类别的预测情况

## 结果

### 模型训练结果

训练完成后，模型将保存在 `models/fusion/` 目录中，包括：
- `best_fusion_model.pth`：性能最佳的模型
- `final_fusion_model.pth`：最终训练模型
- `training_curves.png`：训练和验证损失曲线
- `confusion_matrix.png`：测试集上的混淆矩阵

### 应用运行结果

运行Streamlit应用后，您可以：
- 通过聊天界面与系统进行实时对话
- 查看系统识别的情绪状态和置信度
- 接收基于情绪的共情回应
- 监控系统状态和资源使用情况

## 注意事项

1. **数据路径**：确保 DATA_DIR 指向正确的数据集路径
2. **依赖项**：确保所有依赖项已正确安装
3. **硬件要求**：训练过程可能需要GPU加速
4. **数据预处理**：首次运行时需要较长时间进行数据预处理

## 扩展建议

1. **数据增强**：添加数据增强技术提高模型泛化能力
2. **模型优化**：尝试不同的融合策略和模型结构
3. **超参数调优**：使用网格搜索或贝叶斯优化调优超参数
4. **更多模态**：考虑添加其他模态数据，如文本或姿态

## 许可证

本项目使用 MIT 许可证。

## 联系方式

如有问题或建议，请联系项目维护者。
