import opensmile
import pandas as pd
import os

# YOUR EXACT PATH
BASE_PATH = r"D:\ISEF Model\Dataset\Spanish"

def get_correct_audio_file(folder_path, base_name):
    """Smart file selection: _participant if exists, else full file"""
    participant_file = os.path.join(folder_path, f"{base_name}_participant.wav")
    full_file = os.path.join(folder_path, f"{base_name}.wav")
    
    if os.path.exists(participant_file):
        print(f"  ✅ Using participant: {base_name}_participant.wav")
        return participant_file
    elif os.path.exists(full_file):
        print(f"  ✅ Using full: {base_name}.wav (no participant file)")
        return full_file
    else:
        print(f"  ❌ Missing: {base_name}.wav or _participant.wav")
        return None

# Initialize openSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

features_list = []
print("🔬 Extracting acoustic features from train files...")

# Process AD folder
ad_folder = os.path.join(BASE_PATH, 'english', 'AD')
for file in os.listdir(ad_folder):
    if file.endswith('.wav') and not file.endswith('_participant.wav'):  # Base files only
        base_name = file.replace('.wav', '')
        audio_path = get_correct_audio_file(ad_folder, base_name)
        if audio_path:
            print(f"AD: Processing {os.path.basename(audio_path)}")
            features = smile.process_file(audio_path)
            features['filename'] = base_name + '.wav'  # Match your labels
            features_list.append(features)

# Process CN folder  
cn_folder = os.path.join(BASE_PATH, 'english', 'CN')
for file in os.listdir(cn_folder):
    if file.endswith('.wav') and not file.endswith('_participant.wav'):  # Base files only
        base_name = file.replace('.wav', '')
        audio_path = get_correct_audio_file(cn_folder, base_name)
        if audio_path:
            print(f"CN: Processing {os.path.basename(audio_path)}")
            features = smile.process_file(audio_path)
            features['filename'] = base_name + '.wav'  # Match your labels
            features_list.append(features)

# Save NEW acoustic features
acoustic_clean_df = pd.concat(features_list, ignore_index=True)
acoustic_clean_df.to_csv('spanish_acoustic_features.csv', index=False)
print(f"\n🎉 SAVED: spanish_acoustic_features.csv ({len(features_list)} files)")
print("✅ READY for retraining!")

