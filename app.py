import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image
import cv2
import librosa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import EMOTION_CLASSES
from models.fusion_model import AttentionFusion
from llm_integration import get_llm_integration


class StreamlitApp:
    def __init__(self):
        self.llm = get_llm_integration()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def load_model(self):
        try:
            st.info("加载情感识别模型...")
            model_path = os.path.join('models', 'fusion', 'final_fusion_model.pth')  # 或 'best_fusion_model.pth'
            if not os.path.exists(model_path):
                st.error(f"模型文件不存在: {model_path}")
                return None

            # 你的训练输入维度（从日志中获取）
            voice_dim = 340
            face_dim = 1434
            eda_dim = 6

            model = AttentionFusion((voice_dim, face_dim, eda_dim), hidden_dim=256)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            st.success("模型加载成功！")
            return model
        except Exception as e:
            st.error(f"加载模型失败: {e}")
            return None

    def predict_emotion(self, voice_features=None, face_features=None, eda_features=None):
        if self.model is None:
            st.warning("模型未加载，使用模拟预测")
            import random
            emotion = random.choice(EMOTION_CLASSES)
            confidence = random.uniform(0.6, 0.95)
            return emotion, confidence

        try:
            # 确保输入是 tensor 并在设备上
            inputs = []
            if voice_features is not None:
                inputs.append(torch.tensor(voice_features, dtype=torch.float32).unsqueeze(0).to(self.device))
            else:
                inputs.append(torch.zeros(1, 340).to(self.device))  # 填充默认值

            if face_features is not None:
                inputs.append(torch.tensor(face_features, dtype=torch.float32).unsqueeze(0).to(self.device))
            else:
                inputs.append(torch.zeros(1, 1434).to(self.device))

            if eda_features is not None:
                inputs.append(torch.tensor(eda_features, dtype=torch.float32).unsqueeze(0).to(self.device))
            else:
                inputs.append(torch.zeros(1, 6).to(self.device))

            with torch.no_grad():
                logits, _ = self.model(*inputs)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()
                emotion = EMOTION_CLASSES[pred_idx]

            return emotion, confidence
        except Exception as e:
            st.error(f"预测出错: {e}")
            return "neutral", 0.5

    def extract_voice_features(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return np.mean(mfcc, axis=1)  # 你的训练特征维度是340，需要匹配
        except Exception as e:
            st.error(f"语音特征提取失败: {e}")
            return None

    def extract_face_features(self, image_file):
        try:
            image = np.array(Image.open(image_file))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (38, 38))  # 假设你的训练是1434维，调整大小匹配
            return resized.flatten()
        except Exception as e:
            st.error(f"面部特征提取失败: {e}")
            return None

    def run(self):
        st.title("智能陪护系统 - 多模态情绪识别与共情对话")
        st.write("支持文本、语音、图像输入，识别情绪并生成共情回应")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("说点什么吧...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 模拟多模态输入（实际可扩展为摄像头/麦克风）
            emotion, confidence = self.predict_emotion()

            st.info(f"检测到情绪：**{emotion}** (置信度: {confidence:.2%})")

            with st.spinner("思考中..."):
                context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
                response = self.llm.generate_response_with_fallback(emotion, user_input, context)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        # 侧边栏显示状态
        with st.sidebar:
            st.header("系统状态")
            st.write(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            st.write(f"模型: {'已加载' if self.model else '未加载'}")
            st.write(f"LLM: {'API模式' if hasattr(self.llm, 'api_key') else '模拟模式'}")


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()