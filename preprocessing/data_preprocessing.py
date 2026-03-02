import os
import sys
import pandas as pd
import numpy as np
import librosa
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import mediapipe as mp  # pip install mediapipe
from scipy.signal import butter, filtfilt

# 尝试导入 dlib（可选 fallback）
try:
    import dlib
except ImportError:
    dlib = None

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, PROCESSED_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'voice'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'face'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'eda'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'labels'), exist_ok=True)

DATASET_DIR = DATA_DIR  # Sessions 文件夹根路径

# MediaPipe 初始化（推荐替代 dlib）
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def download_dataset():
    print('请手动下载 MAHNOB-HCI 到:', DATASET_DIR)
    print('预期结构: Sessions/ 下每个 Part_xx_S_Trialyy_emotion 文件夹含 .bdf, .avi, .wav, session.xml')

def extract_dataset():
    if not os.path.exists(DATASET_DIR):
        print('警告：路径不存在:', DATASET_DIR)
    else:
        print('数据集就绪。')

# 提取标签从 session.xml（修正版）
def extract_labels():
    print('提取标签...')
    labels = []
    xml_files = []
    
    # 遍历 Sessions 下所有 session.xml
    for root_dir, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower() == 'session.xml':
                xml_files.append(os.path.join(root_dir, file))

    print(f"找到 {len(xml_files)} 个 session.xml 文件")

    for xml_path in tqdm(xml_files, desc="处理 XML"):
        try:
            tree = ET.parse(xml_path)
            session_root = tree.getroot()
            
            # trial_id：使用文件夹名（更可靠）
            trial_id = os.path.basename(os.path.dirname(xml_path))
            
            # 从子元素中提取（不是属性！）
            valence_elem = session_root.find('.//valence') or session_root.find('.//feltVlnc')
            arousal_elem = session_root.find('.//arousal') or session_root.find('.//feltArsl')
            emotion_elem = session_root.find('.//feltEmo') or session_root.find('.//emotion')
            control_elem = session_root.find('.//feltCtrl') or session_root.find('.//control')
            predictability_elem = session_root.find('.//feltPred') or session_root.find('.//predictability')
            
            # 安全转换为 float/int，缺失则 np.nan
            valence = float(valence_elem.text) if valence_elem is not None and valence_elem.text else np.nan
            arousal = float(arousal_elem.text) if arousal_elem is not None and arousal_elem.text else np.nan
            emotion = int(emotion_elem.text) if emotion_elem is not None and emotion_elem.text else np.nan
            control = float(control_elem.text) if control_elem is not None and control_elem.text else np.nan
            predictability = float(predictability_elem.text) if predictability_elem is not None and predictability_elem.text else np.nan
            
            # 可选：提取 experimentType、mediaFile 等元信息
            exp_type = session_root.get('experimentType', 'unknown')
            media_file = session_root.get('mediaFile', 'unknown')
            
            labels.append({
                'trial_id': trial_id,
                'valence': valence,
                'arousal': arousal,
                'emotion': emotion,
                'control': control,
                'predictability': predictability,
                'experiment_type': exp_type,
                'media_file': media_file,
                'xml_path': xml_path  # 调试用，可选
            })
            
        except ET.ParseError as e:
            print(f"XML 解析失败 {xml_path}: {e}")
        except Exception as e:
            print(f"处理失败 {xml_path}: {e}")

    if labels:
        df = pd.DataFrame(labels)
        # 过滤掉完全没有情感标签的行（可选，根据需要）
        df_valid = df.dropna(subset=['valence', 'arousal'], how='all')
        df.to_csv(os.path.join(PROCESSED_DIR, 'labels_all.csv'), index=False)
        df_valid.to_csv(os.path.join(PROCESSED_DIR, 'labels_valid.csv'), index=False)
        
        print(f"总记录: {len(df)} 条")
        print(f"有效情感记录（至少有 valence 或 arousal）: {len(df_valid)} 条")
        print(f"保存路径: {PROCESSED_DIR}/labels_all.csv 和 labels_valid.csv")
    else:
        print("没有找到任何有效标签！请检查 Sessions 路径和 XML 文件。")

# 语音预处理（基本不变，优化了维度）
def preprocess_voice():
    print('预处理语音...')
    voice_files = [os.path.join(root, f) for root, _, files in os.walk(DATASET_DIR) 
                   for f in files if f.endswith('.wav')]

    print(f'找到 {len(voice_files)} 个 .wav 文件')

    for vf in tqdm(voice_files):
        try:
            y, sr = librosa.load(vf, sr=16000)
            # MFCC + deltas (78 dim)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_feats = np.hstack([np.mean(mfcc,1), np.std(mfcc,1), np.mean(delta,1), np.std(delta,1), np.mean(delta2,1), np.std(delta2,1)])

            # Mel (256 dim)
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
            mel_feats = np.hstack([np.mean(mel,1), np.std(mel,1)])

            # F0, energy, spectral (简化)
            f0, _ = librosa.piptrack(y=y, sr=sr)
            f0 = f0[f0 > 0]
            f0_feats = np.array([np.mean(f0) if len(f0)>0 else 0, np.std(f0) if len(f0)>0 else 0])
            rmse = librosa.feature.rms(y=y)[0]
            energy_feats = np.array([np.mean(rmse), np.std(rmse)])
            spec_feats = np.hstack([
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr),1),
                np.std(librosa.feature.spectral_centroid(y=y, sr=sr),1),
                # ... 可加其他 spectral
            ])

            all_feats = np.concatenate([mfcc_feats, mel_feats, f0_feats, energy_feats, spec_feats])
            out = os.path.join(PROCESSED_DIR, 'voice', os.path.basename(vf).replace('.wav', '.npy'))
            np.save(out, all_feats)
        except Exception as e:
            print(f'语音错误 {vf}: {e}')

# 面部预处理（优先 MediaPipe）
def preprocess_face():
    print('预处理面部...')
    face_files = [os.path.join(root, f) for root, _, files in os.walk(DATASET_DIR) 
                  for f in files if f.endswith('.avi') and 'C1' in f]  # 优先 C1 彩色正面

    print(f'找到 {len(face_files)} 个 C1 .avi 文件')

    for ff in tqdm(face_files):
        try:
            cap = cv2.VideoCapture(ff)
            landmarks_list = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]
                    coords = []
                    for lm_point in lm.landmark:
                        coords.extend([lm_point.x, lm_point.y, lm_point.z])  # 468 points * 3 = 1404 dim per frame
                    landmarks_list.append(coords)
                frame_count += 1
                if frame_count > 300: break  # 限制帧数，避免太慢

            cap.release()

            if landmarks_list:
                mean_lm = np.mean(landmarks_list, axis=0)
                out = os.path.join(PROCESSED_DIR, 'face', os.path.basename(ff).replace('.avi', '.npy'))
                np.save(out, mean_lm)
                print(f'面部特征保存: {out} (dim={len(mean_lm)})')
        except Exception as e:
            print(f'面部错误 {ff}: {e}')

# EDA/GSR 预处理（支持 .bdf 和 .txt）
def preprocess_eda():
    print('预处理皮肤电...')
    eda_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith('.bdf') or (f.endswith('.txt') and ('gsr' in f.lower() or 'eda' in f.lower() or 'skin' in f.lower())):
                eda_files.append(os.path.join(root, f))

    print(f'找到 {len(eda_files)} 个 EDA 文件')

    for ef in tqdm(eda_files):
        try:
            if ef.endswith('.bdf'):
                # 用 mne 读取（pip install mne）
                import mne
                raw = mne.io.read_raw_bdf(ef, preload=True)
                # GSR 通常在 channel 'GSR1' 或 EXG7/EXG8，需检查
                possible_gsr_channels = ['GSR1', 'GSR2', 'EDA', 'EXG7', 'EXG8']
                gsr_channel = next((ch for ch in possible_gsr_channels if ch in raw.ch_names), None)
                if gsr_channel:
                    eda_signal = raw.get_data(picks=gsr_channel)[0]
                else:
                    print(f'未找到 GSR channel in {ef}, 跳过')
                    continue
                fs = raw.info['sfreq']
            else:
                # 原 .txt
                df = pd.read_csv(ef, sep='\t', header=None)
                eda_signal = df.iloc[:,0].values
                fs = 512  # 假设 Biosemi 采样率，实际查 xml 或手动设

            # 预处理：低通滤波 0-5Hz（典型 EDA）
            b, a = butter(4, 5 / (fs / 2), btype='low')
            eda_clean = filtfilt(b, a, eda_signal)

            # 特征提取（扩展）
            mean = np.mean(eda_clean)
            std = np.std(eda_clean)
            # ... 其他原特征
            # 可加 NeuroKit2 scr_peaks 等，但保持简单
            features = np.array([mean, std, np.max(eda_clean), np.min(eda_clean), pd.Series(eda_clean).skew(), pd.Series(eda_clean).kurtosis()])  # 扩展

            out = os.path.join(PROCESSED_DIR, 'eda', os.path.basename(ef).replace('.bdf', '.npy').replace('.txt', '.npy'))
            np.save(out, features)
        except Exception as e:
            print(f'EDA 错误 {ef}: {e}')

def main():
    download_dataset()
    extract_dataset()
    extract_labels()       # 新增：标签
    preprocess_voice()
    preprocess_face()
    preprocess_eda()
    print('预处理完成！特征在 PROCESSED_DIR，标签在 labels.csv')

if __name__ == '__main__':
    main()