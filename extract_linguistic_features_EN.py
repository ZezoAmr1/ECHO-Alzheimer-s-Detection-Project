import whisper
import os

# ============================================
# CONFIGURATION
# ============================================
BASE_PATH = r"C:\Users\Zezo\Documents\ECHO PARS\Records New"
# Supported audio formats
audio_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.aac')

# ============================================
# LOAD WHISPER
# ============================================
print("\n🔄 Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Whisper loaded")

# ============================================
# TRANSCRIPTION LOOP
# ============================================
all_texts = []
audio_files = [f for f in os.listdir(BASE_PATH) if f.lower().endswith(audio_extensions)]

print(f"🎙️ Found {len(audio_files)} records. Starting transcription...\n")

for i, filename in enumerate(audio_files, 1):
    file_path = os.path.join(BASE_PATH, filename)
    print(f"[{i}/{len(audio_files)}] Transcribing: {filename}...")
    
    try:
        # language='en' forces English and prevents "corrupt" Arabic results
        # fp16=False prevents CPU errors
        result = model.transcribe(file_path, language='en', fp16=False)
        
        # Add the text to our list
        all_texts.append(result['text'].strip())
        
    except Exception as e:
        print(f"❌ Error transcribing {filename}: {e}")

# ============================================
# FINAL CONCATENATED OUTPUT
# ============================================
# Combine all transcripts into one big paragraph
full_paragraph = " ".join(all_texts)

print("\n" + "="*50)
print("FINAL CONCATENATED TRANSCRIPT:")
print("="*50)
print(full_paragraph)
print("="*50)

# Optional: Save it to a file
with open("final_transcript_paragraph.txt", "w", encoding="utf-8") as f:
    f.write(full_paragraph)
    print("\n💾 Saved full paragraph to: final_transcript_paragraph.txt")