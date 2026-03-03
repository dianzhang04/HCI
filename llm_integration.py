import os
import sys
import time
import requests
from typing import Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import API_KEY, API_URL, MODEL_NAME
from emotion_prompts import EmotionPromptGenerator

class LLMIntegration:
    def __init__(self):
        self.api_key = API_KEY
        self.api_url = API_URL
        self.model_name = MODEL_NAME
        self.prompt_generator = EmotionPromptGenerator()
        self.max_retries = 3
        self.retry_delay = 1

    def generate_empathetic_response(self, emotion: str, confidence: float, user_input: str, context: str = "") -> Tuple[Optional[str], Optional[Dict]]:
        try:
            prompt = self.prompt_generator.generate_responsive_prompt(
                emotion=emotion,
                confidence=confidence,
                context=context
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]

            response, info = self._call_api(messages)
            return response, info

        except Exception as e:
            print(f"生成回应出错: {e}")
            return None, {"error": str(e)}

    def _call_api(self, messages: list) -> Tuple[Optional[str], Dict]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300,
            "top_p": 0.9
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    verify=False  # 临时关闭验证，只用于调试
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                info = {
                    "model": data.get("model"),
                    "usage": data.get("usage"),
                    "finish_reason": data["choices"][0].get("finish_reason")
                }
                return text, info
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None, {"error": str(e)}

    def generate_response_with_fallback(self, emotion: str, user_input: str, context: str = "") -> str:
        response, _ = self.generate_empathetic_response(emotion, 0.7, user_input, context)
        if response:
            return response
        # fallback
        fallbacks = {
            'joy': "哇，听起来你今天很开心！是什么让你这么快乐呀？",
            'sadness': "我能感受到你的难过。如果你愿意说说，我在这里陪着你。",
            'anger': "我理解你现在很生气。慢慢说出来，我听着呢。",
            'anxiety': "看起来你有点担心。没关系，我们一起面对，好吗？"
        }
        return fallbacks.get(emotion, "我在这里，如果你想聊聊，我随时听着。")

class MockLLMIntegration:
    def __init__(self):
        self.prompt_generator = EmotionPromptGenerator()

    def generate_empathetic_response(self, emotion: str, confidence: float, user_input: str, context: str = "") -> Tuple[str, Dict]:
        prompt = self.prompt_generator.generate_responsive_prompt(emotion, confidence, context)
        mock_responses = {
            'joy': "哈哈，太好了！你今天这么开心，我也被感染了！",
            'sadness': "听到你这么说我心里也不好受。抱抱你，有什么我能帮你的吗？",
            'anger': "我完全理解你的愤怒。来，深呼吸，我们慢慢聊。",
            'anxiety': "没事的，焦虑的时候我陪着你。我们一步一步来，好吗？"
        }
        response = mock_responses.get(emotion, "我在呢，想说什么都可以告诉我。")
        return response, {"mock": True, "prompt": prompt}

def get_llm_integration():
    if not API_KEY or API_KEY == "":
        print("未配置API密钥，使用模拟LLM")
        return MockLLMIntegration()
    return LLMIntegration()