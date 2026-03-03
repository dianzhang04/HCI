from config.config import EMOTION_CLASSES, EMOTION_INTENSITY_LEVELS


class EmotionPromptGenerator:
    """情绪到Prompt的映射生成器（适配9类）"""

    def __init__(self):
        # 9类情绪的Prompt模板
        self.emotion_prompts = {
            'sadness': {
                'low': [
                    "用户情绪略微低落，有轻微的悲伤感。请以温柔、理解的语气回应，提供情感支持。",
                    "用户心情有些沉重。请以关怀、陪伴的态度回应，让用户感到被倾听。"
                ],
                'medium': [
                    "用户现在感到悲伤，情绪明显低落。请以深度共情、温暖的语气回应，提供安慰和理解。",
                    "用户处于悲伤状态，需要被支持。请以耐心、非评判的方式回应。"
                ],
                'high': [
                    "用户非常悲伤，情绪极度低落。请以极度温柔、支持的语气回应，给予强烈的情感陪伴。",
                    "用户处于深度悲伤状态。请以安全、包容的方式回应，让用户感受到被保护。"
                ]
            },
            'joy': {
                'low': [
                    "用户略微开心，有轻微的快乐感。请以愉快、积极的语气回应，分享这份小喜悦。",
                    "用户心情稍微向上。请以轻松、鼓励的方式互动。"
                ],
                'medium': [
                    "用户现在心情很好，明显感到快乐。请以热情、欢快的语气回应，放大用户的积极情绪。",
                    "用户处于喜悦状态。请以庆祝、共鸣的态度回应。"
                ],
                'high': [
                    "用户非常开心，情绪高涨。请以充满活力、兴奋的语气回应，与用户一起庆祝。",
                    "用户极度喜悦。请以热情洋溢的方式回应，强化这份幸福感。"
                ]
            },
            'disgust': {
                'low': [
                    "用户有些反感，有轻微厌恶感。请以理解、中立的语气回应，避免评判。",
                    "用户略有排斥。请以尊重、包容的态度互动。"
                ],
                'medium': [
                    "用户明显感到厌恶。请以接纳、理解的语气回应，认可用户的感受。",
                    "用户处于厌恶状态。请以非评判、开放的方式回应。"
                ],
                'high': [
                    "用户极度厌恶。请以极度理解、包容的语气回应，首先认可用户的感受。",
                    "用户处于强烈厌恶状态。请以安全、非对抗的方式回应。"
                ]
            },
            'neutral': {
                'low': [
                    "用户情绪平静，略微中性。请以友好、自然的语气回应。",
                    "用户保持中立态度。请以平和、专业的语气交流。"
                ],
                'medium': [
                    "用户情绪稳定，处于正常状态。请以温暖、真诚的态度回应。",
                    "用户当前中性，没有明显波动。请以开放、包容的语气互动。"
                ],
                'high': [
                    "用户高度平静，心态平和。请以亲切、贴心的语气回应。",
                    "用户处于极度平静状态。请以积极、鼓励的方式交流。"
                ]
            },
            'amusement': {
                'low': [
                    "用户有点开心，感到轻微愉悦。请以轻松、愉快的语气回应。",
                    "用户略微被逗乐。请以有趣、活泼的方式互动。"
                ],
                'medium': [
                    "用户明显感到愉悦。请以热情、欢快的语气回应，分享这份乐趣。",
                    "用户处于娱乐状态。请以庆祝、共鸣的态度回应。"
                ],
                'high': [
                    "用户非常开心，被逗得大笑。请以充满活力、兴奋的语气回应。",
                    "用户极度愉悦。请以热情洋溢的方式一起庆祝。"
                ]
            },
            'anger': {
                'low': [
                    "用户有些烦躁，轻微愤怒。请以冷静、理解的语气回应。",
                    "用户稍微生气。请以平和、理性的方式互动。"
                ],
                'medium': [
                    "用户明显愤怒。请以耐心、共情的语气回应，认可感受。",
                    "用户处于愤怒状态。请以接纳、非对抗的态度回应。"
                ],
                'high': [
                    "用户极度愤怒。请以极度冷静、理解的语气回应，避免激化。",
                    "用户处于强烈愤怒状态。请以安全、包容的方式回应。"
                ]
            },
            'fear': {
                'low': [
                    "用户有些担心，轻微恐惧。请以安心、支持的语气回应。",
                    "用户略微紧张。请以平稳、鼓励的方式互动。"
                ],
                'medium': [
                    "用户明显恐惧。请以温暖、安抚的语气回应，提供安全感。",
                    "用户处于恐惧状态。请以坚定、可靠的态度回应。"
                ],
                'high': [
                    "用户极度害怕。请以温柔、安抚的语气回应，给予强烈支持。",
                    "用户处于深度恐惧状态。请以稳定、安全的方式回应。"
                ]
            },
            'surprise': {
                'low': [
                    "用户有些惊讶，轻微意外。请以好奇、友好的语气回应。",
                    "用户稍微意外。请以轻松、有趣的方式互动。"
                ],
                'medium': [
                    "用户明显惊讶。请以同理、好奇的语气回应，分享这份意外。",
                    "用户处于惊讶状态。请以热情、参与的态度回应。"
                ],
                'high': [
                    "用户极度惊讶。请以充满活力、好奇的语气回应。",
                    "用户处于强烈惊讶状态。请以热情、参与的方式回应。"
                ]
            },
            'anxiety': {
                'low': [
                    "用户有些焦虑，轻微不安。请以安心、支持的语气回应。",
                    "用户略微担心。请以平稳、鼓励的方式互动。"
                ],
                'medium': [
                    "用户明显焦虑。请以温暖、安抚的语气回应，提供安全感。",
                    "用户处于焦虑状态。请以耐心、包容的态度回应。"
                ],
                'high': [
                    "用户极度焦虑。请以温柔、支持的语气回应，给予强烈陪伴。",
                    "用户处于深度焦虑状态。请以稳定、安全的方式回应。"
                ]
            }
        }

        self.general_instructions = """
        作为一个共情的智能陪伴系统，你的回应应该：
        1. 真诚表达对用户情绪的理解和认可
        2. 避免空洞的安慰，提供具体的情感支持
        3. 保持语言自然、口语化，避免生硬的表达
        4. 根据用户的情绪状态调整回应的语气和内容
        5. 在适当的时候提供积极的引导，帮助用户改善情绪
        6. 尊重用户的隐私，不进行过度的情感侵入
        7. 保持专业的态度，同时展现出人性的温暖
        """

    def generate_prompt(self, emotion, intensity='medium'):
        if emotion not in self.emotion_prompts:
            emotion = 'neutral'
        if intensity not in self.emotion_prompts[emotion]:
            intensity = 'medium'

        import random
        prompt_template = random.choice(self.emotion_prompts[emotion][intensity])
        full_prompt = f"{self.general_instructions}\n\n当前用户情绪状态：\n{prompt_template}"
        return full_prompt

    def get_emotion_intensity(self, confidence):
        if confidence < 0.33:
            return 'low'
        elif confidence < 0.66:
            return 'medium'
        else:
            return 'high'

    def generate_responsive_prompt(self, emotion, confidence, context=""):
        intensity = self.get_emotion_intensity(confidence)
        base_prompt = self.generate_prompt(emotion, intensity)

        if context:
            context_prompt = f"\n\n对话上下文：\n{context}\n\n请结合上述上下文，生成符合当前情绪状态的回应。"
            return base_prompt + context_prompt
        return base_prompt


if __name__ == '__main__':
    generator = EmotionPromptGenerator()
    test_cases = [
        ('joy', 0.8, "今天天气真好！"),
        ('sadness', 0.4, "我今天遇到了一些困难。"),
        ('anger', 0.2, "这件事真的让我很生气！"),
        ('anxiety', 0.7, "我有点担心明天的考试。")
    ]
    for emotion, conf, inp in test_cases:
        prompt = generator.generate_responsive_prompt(emotion, conf, inp)
        print(f"\n情绪: {emotion}, 置信度: {conf}")
        print(f"Prompt: {prompt[:200]}...")  # 截断显示