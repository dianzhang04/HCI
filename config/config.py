# 项目配置文件

# 数据配置
DATA_DIR = 'E:\\HCI-Tagging\\HCI-Tagging Databases\\Sessions'
PROCESSED_DIR = 'preprocessing/processed'

# 模型配置
MODEL_DIR = 'models'
VOICE_MODEL_PATH = 'models/voice/voice_model.pth'
FACE_MODEL_PATH = 'models/face/face_model.pth'
EDA_MODEL_PATH = 'models/eda/eda_model.pth'
FUSION_MODEL_PATH = 'models/fusion/fusion_model.pth'

# 训练配置
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# 大模型API配置
API_KEY = ''  # 请在此处填写API密钥
API_URL = 'https://api.openai.com/v1/chat/completions'
MODEL_NAME = 'gpt-3.5-turbo'

# 前端配置
STREAMLIT_PORT = 8501

# 情感类别
EMOTION_CLASSES = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

# 情绪强度级别
EMOTION_INTENSITY_LEVELS = ['low', 'medium', 'high']
