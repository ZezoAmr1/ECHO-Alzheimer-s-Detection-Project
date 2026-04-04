import os
import whisper

# ... (Your existing "LOAD MODELS" and "CONFIGURATION" code goes here) ...

# ============================================
# LOAD MODELS
# ============================================
print("\n🔄 Loading models...")

# Load Whisper for transcription
whisper_model = whisper.load_model("base")
print("✅ Whisper loaded")


# ============================================
# CONFIGURATION
# ============================================
BASE_PATH = r"C:\Users\Zezo\Documents\ECHO PARS\Records New"
# ============================================
# TRANSCRIPTION PROCESS
# ============================================
print(f"\n📂 Scanning directory: {BASE_PATH}")

# List of common audio extensions
audio_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.aac')

# List to store individual transcripts
all_transcripts = []

# Get all audio files and sort them (to ensure order)
audio_files = [f for f in os.listdir(BASE_PATH) if f.lower().endswith(audio_extensions)]

print(f"🎙️ Found {len(audio_files)} audio records. Starting transcription...\n")

for i, filename in enumerate(audio_files, 1):
    file_path = os.path.join(BASE_PATH, filename)
    
    print(f"[{i}/{len(audio_files)}] Transcribing: {filename}...")
    
    try:
        # Perform transcription
        # Note: Set verbose=False to keep the console clean
        result = whisper_model.transcribe(file_path, fp16=False) 
        
        # Append the text to our list
        all_transcripts.append(result['text'].strip())
        
    except Exception as e:
        print(f"❌ Error transcribing {filename}: {e}")

# ============================================
# FINAL CONCATENATION
# ============================================

# Join all transcripts into one paragraph separated by spaces
full_paragraph = " ".join(all_transcripts)

print("\n✅ All transcriptions complete!")
print("-" * 30)
print("FINAL CONCATENATED PARAGRAPH:")
print(full_paragraph)
print("-" * 30)

# Optional: Save to a text file for your Word Cloud generator
with open("combined_transcript.txt", "w", encoding="utf-8") as f:
    f.write(full_paragraph)
    print("💾 Saved to combined_transcript.txt")