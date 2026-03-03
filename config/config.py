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
API_KEY = '7189489b-31e7-4189-9927-44ca88f17145'  # 请在此处填写API密钥
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODEL_NAME = 'doubao-seed-1-6-251015'

# 前端配置
STREAMLIT_PORT = 8501

# 情感类别
EMOTION_CLASSES = [
    'sadness',       # 1 - 悲伤
    'joy',           # 2 - 快乐，幸福
    'disgust',       # 3 - 厌恶
    'neutral',       # 4 - 中性
    'amusement',     # 5 - 娱乐
    'anger',         # 6 - 愤怒
    'fear',          # 7 - 恐惧
    'surprise',      # 8 - 惊喜
    'anxiety'        # 9 - 焦虑
]

# 情绪强度级别
EMOTION_INTENSITY_LEVELS = ['low', 'medium', 'high']
