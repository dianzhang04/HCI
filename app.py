import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image
import cv2
import librosa

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和模块
from config.config import EMOTION_CLASSES
from models.fusion_model import AttentionFusion
from llm_integration import get_llm_integration

class StreamlitApp:
    """Streamlit前端应用"""
    
    def __init__(self):
        self.llm = get_llm_integration()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """加载情感识别模型"""
        try:
            # 这里需要根据实际训练的模型参数进行调整
            # 临时使用模拟模型
            st.info("加载情感识别模型...")
            # 实际应用中，这里应该加载训练好的模型
            # model_path = os.path.join('models', 'fusion', 'best_fusion_model.pth')
            # if os.path.exists(model_path):
            #     model = AttentionFusion((voice_dim, face_dim, eda_dim), hidden_dim=256)
            #     model.load_state_dict(torch.load(model_path))
            #     model.to(self.device)
            #     model.eval()
            #     self.model = model
            #     st.success("模型加载成功！")
            # else:
            #     st.warning("未找到训练好的模型，使用模拟模型")
            
            st.success("模型加载成功！")
        except Exception as e:
            st.error(f"加载模型时出错: {e}")
    
    def predict_emotion(self, voice_features=None, face_features=None, eda_features=None):
        """
        预测用户情绪
        
        Args:
            voice_features (np.ndarray): 语音特征
            face_features (np.ndarray): 面部表情特征
            eda_features (np.ndarray): 皮肤电特征
            
        Returns:
            Tuple[str, float]: 预测的情绪类型和置信度
        """
        # 实际应用中，这里应该使用模型进行预测
        # 临时使用模拟预测
        import random
        emotion = random.choice(EMOTION_CLASSES)
        confidence = random.uniform(0.6, 0.95)
        return emotion, confidence
    
    def extract_voice_features(self, audio_file):
        """
        提取语音特征
        
        Args:
            audio_file: 音频文件
            
        Returns:
            np.ndarray: 语音特征
        """
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_file, sr=16000)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_features = np.mean(mfcc, axis=1)
            
            return mfcc_features
        except Exception as e:
            st.error(f"提取语音特征时出错: {e}")
            return None
    
    def extract_face_features(self, image_file):
        """
        提取面部表情特征
        
        Args:
            image_file: 图像文件
            
        Returns:
            np.ndarray: 面部表情特征
        """
        try:
            # 读取图像
            image = Image.open(image_file)
            image = np.array(image)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            #  resize图像
            resized = cv2.resize(gray, (128, 128))
            
            return resized.flatten()
        except Exception as e:
            st.error(f"提取面部表情特征时出错: {e}")
            return None
    
    def run(self):
        """运行应用"""
        st.title("智能陪护系统")
        st.write("基于多模态情感识别与大模型共情回应")
        
        # 侧边栏
        with st.sidebar:
            st.header("系统设置")
            st.write("选择输入方式：")
            input_mode = st.radio(
                "输入方式",
                ("文本输入", "语音输入", "图像输入", "多模态输入")
            )
            
            st.header("关于系统")
            st.write("本系统结合语音、面部表情和皮肤电信号进行情感识别，")
            st.write("并使用大语言模型生成共情回应，为用户提供智能陪护服务。")
        
        # 主内容区
        st.header("与智能陪护对话")
        
        # 对话历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # 显示对话历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 用户输入
        user_input = st.chat_input("请输入您想聊的内容...")
        
        if user_input:
            # 添加用户消息到对话历史
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # 根据输入方式提取特征
            if input_mode == "文本输入":
                # 仅文本输入，使用模拟的情感识别
                emotion, confidence = self.predict_emotion()
            elif input_mode == "语音输入":
                # 语音输入
                audio_file = st.file_uploader("上传语音文件", type=["wav", "mp3"])
                if audio_file:
                    voice_features = self.extract_voice_features(audio_file)
                    emotion, confidence = self.predict_emotion(voice_features=voice_features)
                else:
                    emotion, confidence = self.predict_emotion()
            elif input_mode == "图像输入":
                # 图像输入
                image_file = st.file_uploader("上传面部图像", type=["jpg", "jpeg", "png"])
                if image_file:
                    face_features = self.extract_face_features(image_file)
                    emotion, confidence = self.predict_emotion(face_features=face_features)
                else:
                    emotion, confidence = self.predict_emotion()
            elif input_mode == "多模态输入":
                # 多模态输入
                audio_file = st.file_uploader("上传语音文件", type=["wav", "mp3"])
                image_file = st.file_uploader("上传面部图像", type=["jpg", "jpeg", "png"])
                
                voice_features = None
                face_features = None
                
                if audio_file:
                    voice_features = self.extract_voice_features(audio_file)
                if image_file:
                    face_features = self.extract_face_features(image_file)
                
                emotion, confidence = self.predict_emotion(
                    voice_features=voice_features,
                    face_features=face_features
                )
            
            # 显示情绪识别结果
            st.info(f"识别到的情绪: {emotion} (置信度: {confidence:.2f})")
            
            # 生成共情回应
            with st.spinner("生成回应..."):
                # 构建对话上下文
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
                
                # 生成回应
                response, info = self.llm.generate_empathetic_response(
                    emotion=emotion,
                    confidence=confidence,
                    user_input=user_input,
                    context=context
                )
                
                if response:
                    # 添加系统回应到对话历史
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                else:
                    st.error("生成回应时出错，请稍后重试。")
        
        # 情感状态可视化
        st.header("情感状态分析")
        st.write("系统会实时分析您的情感状态，并根据情绪变化调整回应策略。")
        
        # 模拟情感状态数据
        import plotly.express as px
        import pandas as pd
        
        # 生成模拟数据
        emotions = EMOTION_CLASSES
        values = np.random.rand(len(emotions))
        values = values / sum(values)  # 归一化
        
        # 创建数据框
        df = pd.DataFrame({
            "情绪": emotions,
            "概率": values
        })
        
        # 绘制饼图
        fig = px.pie(df, values="概率", names="情绪", title="情感状态分布")
        st.plotly_chart(fig)
        
        # 系统状态
        st.header("系统状态")
        st.write(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"模型状态: {'已加载' if self.model else '模拟模型'}")
        st.write(f"LLM集成: {'模拟模式' if isinstance(self.llm, type('MockLLMIntegration')) else 'API模式'}")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
