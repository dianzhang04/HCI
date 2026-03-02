import os
import sys
import time
import requests
from typing import Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和情绪Prompt生成器
from config.config import API_KEY, API_URL, MODEL_NAME
from emotion_prompts import EmotionPromptGenerator

class LLMIntegration:
    """大语言模型集成模块"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.api_url = API_URL
        self.model_name = MODEL_NAME
        self.prompt_generator = EmotionPromptGenerator()
        self.max_retries = 3
        self.retry_delay = 1  # 秒
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.api_key = api_key
    
    def generate_empathetic_response(self, emotion: str, confidence: float, user_input: str, context: str = "") -> Tuple[Optional[str], Optional[Dict]]:
        """
        生成共情回应
        
        Args:
            emotion (str): 用户情绪类型
            confidence (float): 情绪识别的置信度
            user_input (str): 用户的输入文本
            context (str): 对话上下文
            
        Returns:
            Tuple[Optional[str], Optional[Dict]]: 生成的回应和详细信息
        """
        try:
            # 生成情绪相关的Prompt
            prompt = self.prompt_generator.generate_responsive_prompt(
                emotion=emotion,
                confidence=confidence,
                context=context
            )
            
            # 构建完整的对话历史
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            
            # 调用API生成回应
            response, info = self._call_openai_api(messages)
            
            return response, info
            
        except Exception as e:
            print(f"生成共情回应时出错: {e}")
            return None, {"error": str(e)}
    
    def _call_openai_api(self, messages: list) -> Tuple[Optional[str], Dict]:
        """
        调用OpenAI API
        
        Args:
            messages (list): 对话历史
            
        Returns:
            Tuple[Optional[str], Dict]: 生成的回应和详细信息
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                response.raise_for_status()  # 检查HTTP错误
                
                # 解析响应
                data = response.json()
                generated_text = data["choices"][0]["message"]["content"]
                
                # 提取详细信息
                info = {
                    "model": data.get("model"),
                    "usage": data.get("usage"),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                    "total_tokens": data.get("usage", {}).get("total_tokens")
                }
                
                return generated_text, info
                
            except requests.exceptions.RequestException as e:
                print(f"API调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # 指数退避
                else:
                    return None, {"error": f"API调用失败: {str(e)}"}
            
            except Exception as e:
                print(f"处理API响应时出错: {e}")
                return None, {"error": f"处理API响应时出错: {str(e)}"}
    
    def generate_response_with_fallback(self, emotion: str, user_input: str, context: str = "") -> str:
        """
        生成回应，带有降级策略
        
        Args:
            emotion (str): 用户情绪类型
            user_input (str): 用户的输入文本
            context (str): 对话上下文
            
        Returns:
            str: 生成的回应
        """
        # 尝试使用API生成回应
        response, info = self.generate_empathetic_response(
            emotion=emotion,
            confidence=0.7,  # 默认置信度
            user_input=user_input,
            context=context
        )
        
        # 如果API调用失败，使用预设的回应
        if not response:
            response = self._get_fallback_response(emotion, user_input)
        
        return response
    
    def _get_fallback_response(self, emotion: str, user_input: str) -> str:
        """
        获取降级回应
        
        Args:
            emotion (str): 用户情绪类型
            user_input (str): 用户的输入文本
            
        Returns:
            str: 预设的回应
        """
        fallback_responses = {
            'happy': "我很高兴看到你这么开心！能分享一下是什么让你这么快乐的吗？",
            'sad': "我能感受到你现在心情不太好。如果你愿意分享，我在这里倾听。",
            'angry': "我理解你现在感到生气。有时候表达出来会好一些，我在这里听你说。",
            'fearful': "看起来你有些担心。记住，无论发生什么，我都会在这里支持你。",
            'disgusted': "我理解你现在的感受。每个人都有不喜欢的东西，这很正常。",
            'surprised': "哇，这真的很令人惊讶！能告诉我更多关于这件事的情况吗？",
            'neutral': "谢谢你的分享。我在这里，如果你想聊点什么。"
        }
        
        return fallback_responses.get(emotion, fallback_responses['neutral'])

class MockLLMIntegration:
    """模拟的大语言模型集成（用于开发和测试）"""
    
    def __init__(self):
        self.prompt_generator = EmotionPromptGenerator()
    
    def generate_empathetic_response(self, emotion: str, confidence: float, user_input: str, context: str = "") -> Tuple[str, Dict]:
        """
        生成模拟的共情回应
        
        Args:
            emotion (str): 用户情绪类型
            confidence (float): 情绪识别的置信度
            user_input (str): 用户的输入文本
            context (str): 对话上下文
            
        Returns:
            Tuple[str, Dict]: 生成的回应和详细信息
        """
        # 生成情绪相关的Prompt
        prompt = self.prompt_generator.generate_responsive_prompt(
            emotion=emotion,
            confidence=confidence,
            context=context
        )
        
        # 生成模拟的回应
        mock_responses = {
            'happy': "我能感受到你的快乐！这真是太好了，继续保持这种积极的状态吧！",
            'sad': "我理解你现在的感受，感到悲伤是很正常的。如果你需要倾诉，我就在这里。",
            'angry': "我能理解你的愤怒，这种感觉一定很难受。深呼吸，慢慢说，我在听。",
            'fearful': "不要害怕，无论发生什么，我们都会一起面对。你不是一个人。",
            'disgusted': "我理解你对这件事的感受，每个人都有自己的喜好和界限。",
            'surprised': "哇，这确实令人惊讶！能告诉我更多细节吗？",
            'neutral': "谢谢你的分享。我在这里，如果你想聊点什么。"
        }
        
        response = mock_responses.get(emotion, mock_responses['neutral'])
        
        info = {
            "model": "mock-llm",
            "prompt": prompt,
            "user_input": user_input,
            "context": context,
            "emotion": emotion,
            "confidence": confidence
        }
        
        return response, info
    
    def generate_response_with_fallback(self, emotion: str, user_input: str, context: str = "") -> str:
        """
        生成模拟的回应
        
        Args:
            emotion (str): 用户情绪类型
            user_input (str): 用户的输入文本
            context (str): 对话上下文
            
        Returns:
            str: 生成的回应
        """
        response, _ = self.generate_empathetic_response(
            emotion=emotion,
            confidence=0.7,
            user_input=user_input,
            context=context
        )
        return response

# 工厂函数，根据配置返回合适的LLM集成

def get_llm_integration() -> LLMIntegration:
    """
    获取LLM集成实例
    
    Returns:
        LLMIntegration: LLM集成实例
    """
    # 如果没有API密钥，使用模拟的集成
    if not API_KEY or API_KEY == "":
        print("未配置API密钥，使用模拟的LLM集成")
        return MockLLMIntegration()
    else:
        return LLMIntegration()

if __name__ == '__main__':
    # 测试LLM集成
    llm = get_llm_integration()
    
    # 测试不同情绪的回应生成
    test_cases = [
        ('happy', "今天天气真好！"),
        ('sad', "我今天遇到了一些困难。"),
        ('angry', "这件事真的让我很生气！"),
        ('neutral', "我今天过得还可以。")
    ]
    
    for emotion, user_input in test_cases:
        print(f"\n测试情绪: {emotion}")
        print(f"用户输入: {user_input}")
        
        response, info = llm.generate_empathetic_response(
            emotion=emotion,
            confidence=0.8,
            user_input=user_input,
            context="这是我们的第一次对话"
        )
        
        print(f"生成的回应: {response}")
        print(f"详细信息: {info}")
