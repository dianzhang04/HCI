import os
import sys
import re
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

# 检查 mne 是否安装（EDA 需要）
try:
    import mne
except ImportError:
    mne = None
    print("警告：未安装 mne 库，EDA .bdf 处理将失败。请运行：pip install mne")

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, PROCESSED_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'voice'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'face'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'eda'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'labels'), exist_ok=True)

DATASET_DIR = DATA_DIR  # Sessions 文件夹根路径

# MediaPipe 初始化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def download_dataset():
    print('请手动下载 MAHNOB-HCI 到:', DATASET_DIR)
    print('预期结构: Sessions/ 下每个文件夹（如 1822、10）含 session.xml、.bdf、.avi、.wav 等')


def extract_dataset():
    if not os.path.exists(DATASET_DIR):
        print('警告：路径不存在:', DATASET_DIR)
    else:
        print('数据集路径就绪。')


# ==================== 提取标签（最终版：强制用文件夹名作为 trial_id） ====================
def extract_labels():
    print('正在提取标签（使用 session.xml 属性 + 文件夹名作为 trial_id）...')
    labels = []
    xml_files = []

    for root_dir, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower() == 'session.xml':
                xml_files.append(os.path.join(root_dir, file))

    print(f"找到 {len(xml_files)} 个 session.xml 文件")

    valid_count = 0
    for xml_path in tqdm(xml_files, desc="处理 XML"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 从属性读取（你的 XML 格式）
            valence = float(root.get('feltVlnc')) if root.get('feltVlnc') else np.nan
            arousal = float(root.get('feltArsl')) if root.get('feltArsl') else np.nan
            emotion = int(root.get('feltEmo')) if root.get('feltEmo') else np.nan
            control = float(root.get('feltCtrl')) if root.get('feltCtrl') else np.nan
            predictability = float(root.get('feltPred')) if root.get('feltPred') else np.nan

            # 强制使用文件夹名作为 trial_id（最可靠、最统一）
            trial_id = os.path.basename(os.path.dirname(xml_path))  # 如 "1822"、"10"

            record = {
                'trial_id': trial_id,
                'valence': valence,
                'arousal': arousal,
                'emotion': emotion,
                'control': control,
                'predictability': predictability,
                'experiment_type': root.get('experimentType', 'unknown'),
                'media_file': root.get('mediaFile', 'unknown'),
                'xml_path': xml_path
            }

            labels.append(record)
            if not (np.isnan(valence) and np.isnan(arousal)):
                valid_count += 1

        except Exception as e:
            print(f"XML 处理失败 {xml_path}: {e}")

    if labels:
        df = pd.DataFrame(labels)
        df_all = df.copy()
        df_valid = df.dropna(subset=['valence', 'arousal'], how='all')

        df_all.to_csv(os.path.join(PROCESSED_DIR, 'labels_all.csv'), index=False)
        df_valid.to_csv(os.path.join(PROCESSED_DIR, 'labels_valid.csv'), index=False)

        print(f"\n✅ 标签提取完成！")
        print(f"   总 session 数      : {len(df_all)}")
        print(f"   有有效情感标签数 : {len(df_valid)} ({valid_count} 条)")
        print(f"   保存文件：")
        print(f"      labels_all.csv     ← 所有 session（含无标签）")
        print(f"      labels_valid.csv   ← **推荐训练使用**（只保留有标签的）")

        print("\n前5条有效标签预览（trial_id 示例）：")
        print(df_valid[['trial_id', 'valence', 'arousal']].head())
    else:
        print("❌ 没有找到任何 XML！请检查 Sessions 路径")


# ==================== 语音预处理（文件名带 trial_id 前缀） ====================
def preprocess_voice():
    print('预处理语音...')
    voice_files = [os.path.join(root, f) for root, _, files in os.walk(DATASET_DIR)
                   for f in files if f.endswith('.wav')]

    print(f'找到 {len(voice_files)} 个 .wav 文件')

    for vf in tqdm(voice_files):
        try:
            # 获取 trial_id（文件夹名）
            trial_dir = os.path.dirname(vf)
            trial_id = os.path.basename(trial_dir)

            y, sr = librosa.load(vf, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_feats = np.hstack(
                [np.mean(mfcc, 1), np.std(mfcc, 1), np.mean(delta, 1), np.std(delta, 1), np.mean(delta2, 1),
                 np.std(delta2, 1)])

            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
            mel_feats = np.hstack([np.mean(mel, 1), np.std(mel, 1)])

            f0, _ = librosa.piptrack(y=y, sr=sr)
            f0 = f0[f0 > 0]
            f0_feats = np.array([np.mean(f0) if len(f0) > 0 else 0, np.std(f0) if len(f0) > 0 else 0])
            rmse = librosa.feature.rms(y=y)[0]
            energy_feats = np.array([np.mean(rmse), np.std(rmse)])
            spec_feats = np.hstack([
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), 1),
                np.std(librosa.feature.spectral_centroid(y=y, sr=sr), 1),
            ])

            all_feats = np.concatenate([mfcc_feats, mel_feats, f0_feats, energy_feats, spec_feats])

            # 保存时带 trial_id 前缀
            base_name = os.path.basename(vf).replace('.wav', '')
            out = os.path.join(PROCESSED_DIR, 'voice', f"{trial_id}_{base_name}.npy")
            np.save(out, all_feats)
            print(f"语音特征保存: {out}")

        except Exception as e:
            print(f'语音错误 {vf}: {e}')


# ==================== 面部预处理（文件名带 trial_id 前缀） ====================
def preprocess_face():
    print('预处理面部...')

    # 收集所有 .avi 文件
    face_files = [os.path.join(root, f) for root, _, files in os.walk(DATASET_DIR)
                  for f in files if f.endswith('.avi')]

    print(f'找到 {len(face_files)} 个 .avi 文件')

    # 按 trial_id 分组（文件夹名）
    from collections import defaultdict
    trial_videos = defaultdict(list)
    for ff in face_files:
        trial_dir = os.path.dirname(ff)
        trial_id = os.path.basename(trial_dir)
        trial_videos[trial_id].append(ff)

    print(f'涉及 {len(trial_videos)} 个 trial，各 trial 视频数：')
    for tid, vids in trial_videos.items():
        print(f'  {tid}: {len(vids)} 个avi文件')

    processed_count = 0

    for trial_id, videos in tqdm(trial_videos.items(), desc='处理 trial'):
        try:
            all_landmarks = []
            total_frames = 0
            sampled_frames = 0
            MAX_SAMPLES_PER_VIDEO = 200  # 每个视频最多采样 200 帧
            SAMPLE_INTERVAL = 5  # 每 5 帧采样一次

            for vid_idx, ff in enumerate(videos, 1):
                cap = cv2.VideoCapture(ff)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    total_frames += 1

                    # 采样策略：每 SAMPLE_INTERVAL 帧取一次，且不超过 MAX_SAMPLES_PER_VIDEO
                    if frame_count % SAMPLE_INTERVAL == 0 and sampled_frames < MAX_SAMPLES_PER_VIDEO * len(videos):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(frame_rgb)

                        if results.multi_face_landmarks:
                            lm = results.multi_face_landmarks[0]
                            coords = [coord for lm_point in lm.landmark for coord in (lm_point.x, lm_point.y, lm_point.z)]
                            all_landmarks.append(coords)
                            sampled_frames += 1

                    frame_count += 1

                cap.release()

            tqdm.write(f'trial {trial_id} - 总帧数: {total_frames} | 采样有效帧: {len(all_landmarks)}')

            if all_landmarks:
                mean_lm = np.mean(all_landmarks, axis=0)
                out_path = os.path.join(PROCESSED_DIR, 'face', f"{trial_id}_fused.npy")
                np.save(out_path, mean_lm)
                tqdm.write(f'✅ 保存融合特征: {out_path} (dim={len(mean_lm)})')
                processed_count += 1
            else:
                tqdm.write(f'⚠️ trial {trial_id} 未提取到任何面部关键点')

        except Exception as e:
            tqdm.write(f'❌ trial {trial_id} 处理错误: {e}')
            continue

    print(f"\n面部预处理完成，共生成 {processed_count} 个融合特征文件")

# ==================== EDA/GSR 预处理（文件名带 trial_id 前缀） ====================
def preprocess_eda():
    print('预处理皮肤电...')
    eda_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith('.bdf') or (
                    f.endswith('.txt') and ('gsr' in f.lower() or 'eda' in f.lower() or 'skin' in f.lower())):
                eda_files.append(os.path.join(root, f))

    print(f'找到 {len(eda_files)} 个 EDA 文件')

    for ef in tqdm(eda_files):
        try:
            # 获取 trial_id
            trial_dir = os.path.dirname(ef)
            trial_id = os.path.basename(trial_dir)

            if ef.endswith('.bdf'):
                if mne is None:
                    print(f"跳过 {ef}：缺少 mne 库")
                    continue
                raw = mne.io.read_raw_bdf(ef, preload=True)
                possible_gsr_channels = ['GSR1', 'GSR2', 'EDA', 'EXG7', 'EXG8']
                gsr_channel = next((ch for ch in possible_gsr_channels if ch in raw.ch_names), None)
                if gsr_channel:
                    eda_signal = raw.get_data(picks=gsr_channel)[0]
                else:
                    print(f'未找到 GSR channel in {ef}, 跳过')
                    continue
                fs = raw.info['sfreq']
            else:
                df = pd.read_csv(ef, sep='\t', header=None)
                eda_signal = df.iloc[:, 0].values
                fs = 512  # 假设

            # 低通滤波
            b, a = butter(4, 5 / (fs / 2), btype='low')
            eda_clean = filtfilt(b, a, eda_signal)

            features = np.array([
                np.mean(eda_clean), np.std(eda_clean),
                np.max(eda_clean), np.min(eda_clean),
                pd.Series(eda_clean).skew(), pd.Series(eda_clean).kurtosis()
            ])

            base_name = os.path.basename(ef).replace('.bdf', '').replace('.txt', '')
            out = os.path.join(PROCESSED_DIR, 'eda', f"{trial_id}_{base_name}.npy")
            np.save(out, features)
            print(f"EDA 特征保存: {out}")

        except Exception as e:
            print(f'EDA 错误 {ef}: {e}')


def main():
    download_dataset()
    extract_dataset()
    #extract_labels()  # 必须先跑这个，生成 labels_valid.csv
    #preprocess_voice()   # 取消注释以运行
    preprocess_face()    # 取消注释以运行
    #preprocess_eda()     # 取消注释以运行
    print('\n预处理完成！')
    print(' - 标签文件：processed_io/labels_valid.csv（推荐训练使用）')
    print(' - 特征文件：processed_io/voice/face/eda/（文件名已带 trial_id 前缀，如 1822_xxx.npy）')
    print('下一步：运行 train_fusion.py，使用 labels_valid.csv')


if __name__ == '__main__':
    main()