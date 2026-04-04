import opensmile
import pandas as pd
import os
import tempfile
from pydub import AudioSegment
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# YOUR EXACT PATH
BASE_PATH = r"D:\ISEF Model\Dataset\Spanish"

# ============================================
# MP3 CONVERSION HELPERS (From Spanish Pipeline)
# ============================================

def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to WAV using pydub"""
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav_path = temp_wav.name
        temp_wav.close()
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print(f"    ❌ MP3 conversion failed: {e}")
        return None

def get_correct_audio_file(folder_path, base_name):
    """Smart file selection: handles WAV and MP3 in order of preference"""
    # Check for files in this specific order
    checks = [
        (f"{base_name}_participant.wav", False),
        (f"{base_name}.wav", False),
        (f"{base_name}_participant.mp3", True),
        (f"{base_name}.mp3", True)
    ]
    
    for filename, is_mp3 in checks:
        path = os.path.join(folder_path, filename)
        if os.path.exists(path):
            if is_mp3:
                print(f"    🔄 Converting MP3 to WAV...")
                return convert_mp3_to_wav(path), True # (path, is_temporary_flag)
            else:
                print(f"    ✅ Using WAV: {filename}")
                return path, False
                
    print(f"    ❌ Missing: {base_name}.wav or .mp3 versions")
    return None, False

# Initialize openSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

features_list = []
print("🔬 Extracting acoustic features (MP3 support added)...")

# Process AD folder
ad_folder = os.path.join(BASE_PATH, 'AD')
if os.path.exists(ad_folder):
    for file in os.listdir(ad_folder):
        # Base files only (WAV or MP3)
        if (file.endswith('.wav') or file.endswith('.mp3')) and not file.endswith('_participant.wav') and not file.endswith('_participant.mp3'):
            base_name = file.replace('.wav', '').replace('.mp3', '')
            audio_path, is_temp = get_correct_audio_file(ad_folder, base_name)
            
            if audio_path:
                print(f"AD: Processing {base_name}")
                try:
                    features = smile.process_file(audio_path)
                    features['filename'] = base_name + '.wav'
                    features_list.append(features)
                finally:
                    # Clean up temporary WAV file if it was converted from MP3
                    if is_temp and os.path.exists(audio_path):
                        os.remove(audio_path)

# Process CN folder  
cn_folder = os.path.join(BASE_PATH, 'CN')
if os.path.exists(cn_folder):
    for file in os.listdir(cn_folder):
        if (file.endswith('.wav') or file.endswith('.mp3')) and not file.endswith('_participant.wav') and not file.endswith('_participant.mp3'):
            base_name = file.replace('.wav', '').replace('.mp3', '')
            audio_path, is_temp = get_correct_audio_file(cn_folder, base_name)
            
            if audio_path:
                print(f"CN: Processing {base_name}")
                try:
                    features = smile.process_file(audio_path)
                    features['filename'] = base_name + '.wav'
                    features_list.append(features)
                finally:
                    # Clean up temporary WAV file
                    if is_temp and os.path.exists(audio_path):
                        os.remove(audio_path)

# Save NEW acoustic features
if features_list:
    acoustic_clean_df = pd.concat(features_list, ignore_index=True)
    acoustic_clean_df.to_csv('spanish_acoustic_features.csv', index=False)
    print(f"\n🎉 SAVED: spanish_acoustic_features.csv ({len(features_list)} files)")
else:
    print("\n❌ No features extracted. Check your folder paths.")

print("✅ READY for retraining!")