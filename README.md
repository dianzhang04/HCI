# HCI
店长的毕业论文


# 2026-2-9
河 海 大 学
本科毕业设计（论文）任务书
（理 工 科 类）
Ⅰ、毕业设计（论文）题目：
基于多模态情感识别与大模型共情回应的智能陪护系统研究                                                                    
Ⅱ、毕业设计（论文）工作内容（从综合运用知识、研究方案的设计、研究方法和手段的运用、应用文献资料、数据分析处理、图纸质量、技术或观点创新等方面详细说明）：
在数字化与智能化快速发展的背景下，情感计算与心理健康关怀逐渐成为人机交互领域的重要研究方向。传统的人机交互系统多以指令执行为主，缺乏对用户情绪状态的感知与回应能力，难以在长时间交互中提供真正有温度的陪伴体验。多模态情感识别技术通过融合语音、面部表情、皮肤电等多源信号，可以更全面地刻画用户当下的情绪状态与紧张/放松水平，为智能陪护系统提供可靠的情感感知基础。本课题以语音、面部表情和皮肤电为主要模态，设计并实现一个集多模态情感识别与大模型共情陪护于一体的原型系统。本研究重点围绕以下几方面展开：1）多模态情感识别模型设计：构建语音情感特征、面部表情特征和皮肤电情绪特征的提取网络，并探索基于注意力机制或加权融合的多模态特征融合方法，提高情感识别的准确性与鲁棒性；2）情绪–共情文本生成机制：基于识别到的情绪类别与强度，设计情感状态到Prompt模板的映射策略，调用大模型API 生成上下文相关、语气合适、风格柔和的共情回应文本。
具体实施步骤：1）问题定义与研究需求分析：明确系统面向的应用场景（如情绪舒缓陪伴、学习压力缓解、简单心理关怀等）；分析多模态情感识别在真实场景中面临的挑战，包括背景噪声、表情不完整、皮肤电噪声等；设计总体技术架构：多模态采集模块、情感识别模块、大模型共情回应模块、前端展示模块。2）数据集预处理：选用公开多模态情绪数据集MAHNOB-HCI Dataset，对其进行相应的预处理，确保训练数据的有效性。3）情感特征提取与建模：可对不同模态运用不同模型提取特征；设计多模态融合层；使用交叉熵损失、F1-score、Macro-F1 等指标评估模型在验证集上的性能，并通过超参数调优（学习率、batch size、网络深度等）提升识别效果。4）基于大模型的共情陪护策略设计与实现：设计情绪状态到提示模板的映射规则， 将识别到的情绪类别、强度和部分上下文信息组织成结构化提示，调用大模型API生成自然语言共情回应；设计基本的安全与约束规则，例如限制输出风格为温和、中立、不提供医疗诊断，避免产生负向暗示。5）系统集成：使用Python、深度学习框架（如 PyTorch/TensorFlow）实现后端情感识别模型；使用Web前端或桌面应用框架（如Flask/Streamlit/PyQt）构建简单交互界面，实现情绪可视化与共情文字展示。6）论文撰写与总结：总结研究成果，分析实验结果及其应用意义，撰写完整毕业论文，并提出未来优化方向。                                                                                 
Ⅲ、进度安排：
第八学期　                                                              
第1—3周：相关文献调研与基础知识学习                                      
第3—7周：课题核心方法设计与分析                                                                               
第8—12周：实验验证                                                       
第13—14周：撰写毕业论文                                                                   
第15周：资料整理、代码打包、准备答辩                                                                    
Ⅳ、主要参考资料：
[1]Gladys A A, Vetriselvi V. Survey on multimodal approaches to emotion recognition[J]. Neurocomputing, 2023, 556: 126693.                                                  
[2]Pan B, Hirota K, Jia Z, et al. A review of multimodal emotion recognition from datasets, preprocessing, features, and fusion methods[J]. Neurocomputing, 2023, 561: 126866.                                                                              
[3]Hu D, Hou X, Wei L, et al. MM-DFN: Multimodal dynamic fusion network for emotion recognition in conversations[C]//ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 7037-7041.                                                      
[4]Li D, Wang Y, Funakoshi K, et al. Joyful: Joint modality fusion and graph contrastive learning for multimodal emotion recognition[J]. arXiv preprint arXiv:2311.11009, 2023.                                                          
[5]Ouzar Y, Bousefsaf F, Djeldjli D, et al. Video-based multimodal spontaneous emotion recognition using facial expressions and physiological signals[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 2460-2469.                                                
[6]Wiem M B H, Lachiri Z. Emotion classification in arousal valence model using MAHNOB-HCI database[J]. International Journal of Advanced Computer Science and Applications, 2017, 8(3).                                           
[7]Soleymani M, Lichtenauer J, Pun T, et al. A multi-modal affective database for affect recognition and implicit tagging. Affective Computing[J]. IEEE Transactions on Affective Computing, 3: 1-1.                                      
[8]Al Machot F, Elmachot A, Ali M, et al. A deep-learning model for subject-independent human emotion recognition using electrodermal activity sensors[J]. Sensors, 2019, 19(7): 1659.                                      
[9]Ramakrishnan S. Recognition of emotion from speech: A review[J]. Speech Enhancement, Modeling and recognition–algorithms and Applications, 2012, 7: 121-137.                                                   
[10]Yadav S P, Zaidi S, Mishra A, et al. Survey on machine learning in speech emotion recognition and vision systems using a recurrent neural network (RNN)[J]. Archives of Computational Methods in Engineering, 2022, 29(3): 1753-1770.                              