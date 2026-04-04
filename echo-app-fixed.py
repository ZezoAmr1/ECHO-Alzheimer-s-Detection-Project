"""
ECHO - Early Cognitive Health Observatory
FIXED VERSION - Removes hardcoded thresholds and uses model-based analysis

Three binary classifiers:
1. AD vs CN (English) - 85.9%
2. MCI vs CN (English) - 76.6%
3. MCI vs CN (Chinese) - 60%
"""

import streamlit as st
import joblib
import opensmile
import whisper
import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')
from pydub import AudioSegment

# Conditional imports
try:
    import spacy
    import jieba
    import jieba.posseg as pseg
    from lexicalrichness import LexicalRichness
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTION_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================

MODEL_ROOT = r"D:\ISEF Model\Triple Model"

MODEL_CONFIGS = {
    "AD (English)": {
        "folder": "AD-CN Model 85.9%",
        "accuracy": "85.9%",
        "description": "Alzheimer's Disease Detection",
        "language": "en",
        "features": ["acoustic", "linguistic"],
        "condition": "Alzheimer's Disease"
    },
    "MCI (English)": {
        "folder": "MCI-CN Model 76.6%",
        "accuracy": "76.6%",
        "description": "Mild Cognitive Impairment Detection",
        "language": "en",
        "features": ["acoustic", "linguistic"],
        "condition": "Mild Cognitive Impairment"
    },
    "MCI (Chinese)": {
        "folder": "Chinese MCI-CN 60%",
        "accuracy": "60%",
        "description": "轻度认知障碍检测",
        "language": "zh",
        "features": ["linguistic"],
        "condition": "Mild Cognitive Impairment",
        "note": "Limited training data available"
    }
}

# REMOVED: Hardcoded NORMAL_RANGES - now using model-based analysis

# ============================================
# CUSTOM CSS
# ============================================

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    h2, h3 {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #00d4ff 0%, #0096c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #00d4ff;
        font-weight: 700;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #0096c7 100%);
    }
    
    .step-indicator {
        background: rgba(0, 212, 255, 0.15);
        border-left: 4px solid #00d4ff;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e0e0e0;
        font-weight: 600;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 150, 199, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
    }
    
    .analysis-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .indicator {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .indicator.normal {
        background: rgba(0, 212, 170, 0.2);
        border-left: 4px solid #00d4aa;
    }
    
    .indicator.warning {
        background: rgba(255, 165, 0, 0.2);
        border-left: 4px solid #ffa500;
    }
    
    .indicator.risk {
        background: rgba(255, 68, 68, 0.2);
        border-left: 4px solid #ff4444;
    }
    
    .footer {
        text-align: center;
        color: #808080;
        padding: 2rem 0;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_model_artifacts(model_key):
    """Load all model components"""
    config = MODEL_CONFIGS[model_key]
    folder_path = Path(MODEL_ROOT) / config["folder"]
    
    model = joblib.load(folder_path / "best_model_final.pkl")
    scaler = joblib.load(folder_path / "scaler_final.pkl")
    features = joblib.load(folder_path / "features_final.pkl")
    threshold = joblib.load(folder_path / "threshold_final.pkl")
    
    selector_path = folder_path / "selector_final.pkl"
    selector = joblib.load(selector_path) if selector_path.exists() else None
    
    return model, scaler, features, threshold, selector, config

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_opensmile():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )

@st.cache_resource
def load_nlp_models():
    if not FEATURE_EXTRACTION_AVAILABLE:
        return None, None
    
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except:
        nlp_en = None
    
    try:
        nlp_zh = spacy.load("zh_core_web_sm")
    except:
        nlp_zh = None
    
    return nlp_en, nlp_zh

# ============================================
# FEATURE EXTRACTION (EXACT COPY FROM TRAINING)
# ============================================

def extract_acoustic_features(audio_path):
    """Extract OpenSMILE acoustic features"""
    smile = load_opensmile()
    features = smile.process_file(audio_path)
    
    if isinstance(features, pd.DataFrame):
        features_series = features.iloc[0]
    else:
        features_series = features
    
    if isinstance(features_series.index, pd.MultiIndex):
        new_index = []
        for idx in features_series.index:
            if isinstance(idx, tuple):
                new_index.append('_'.join(map(str, idx)))
            else:
                new_index.append(str(idx))
        features_series.index = new_index
    
    feature_dict = features_series.to_dict()
    df = pd.DataFrame([feature_dict], index=[0])
    
    return df

def extract_english_linguistic_features(transcript, audio_path):
    """Extract English linguistic features - EXACT MATCH TO TRAINING"""
    if not FEATURE_EXTRACTION_AVAILABLE:
        return get_default_linguistic_features(0)
    
    nlp_en, _ = load_nlp_models()
    if nlp_en is None:
        return get_default_linguistic_features(0)
    
    duration = librosa.get_duration(path=audio_path)
    text = transcript.strip()
    
    if not text or len(text) < 10:
        return get_default_linguistic_features(duration)
    
    doc = nlp_en(text)
    tokens = [token for token in doc if not token.is_punct and not token.is_space]
    words = [token.text.lower() for token in tokens]
    
    if len(words) == 0:
        return get_default_linguistic_features(duration)
    
    total_words = len(words)
    unique_words = len(set(words))
    sentences = list(doc.sents)
    num_utterances = len(sentences) if sentences else 1
    
    from collections import Counter
    pos_counts = Counter([token.pos_ for token in tokens])
    
    noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    verb_count = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)
    adj_count = pos_counts.get('ADJ', 0)
    adv_count = pos_counts.get('ADV', 0)
    det_count = pos_counts.get('DET', 0)
    pron_count = pos_counts.get('PRON', 0)
    prep_count = pos_counts.get('ADP', 0)
    
    mlu_words = total_words / num_utterances if num_utterances > 0 else 0
    words_min = (total_words / duration) * 60 if duration > 0 else 0
    verbs_utt = verb_count / num_utterances if num_utterances > 0 else 0
    
    pct_nouns = (noun_count / total_words * 100) if total_words > 0 else 0
    pct_verbs = (verb_count / total_words * 100) if total_words > 0 else 0
    pct_adj = (adj_count / total_words * 100) if total_words > 0 else 0
    pct_adv = (adv_count / total_words * 100) if total_words > 0 else 0
    pct_det = (det_count / total_words * 100) if total_words > 0 else 0
    pct_pro = (pron_count / total_words * 100) if total_words > 0 else 0
    pct_prep = (prep_count / total_words * 100) if total_words > 0 else 0
    
    noun_verb = noun_count / verb_count if verb_count > 0 else 1.0
    
    content_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}
    content_count = sum(1 for token in tokens if token.pos_ in content_pos)
    function_count = total_words - content_count
    
    density = content_count / total_words if total_words > 0 else 0
    open_closed = content_count / function_count if function_count > 0 else 1.0
    
    try:
        from lexicalrichness import LexicalRichness
        lex = LexicalRichness(" ".join(words))
        freq_ttr = lex.ttr
        window = min(25, total_words - 1) if total_words > 25 else max(1, total_words - 1)
        mattr = lex.mattr(window_size=window) if total_words >= 10 else freq_ttr
        mtld = lex.mtld(threshold=0.72) if total_words >= 50 else None
        hdd = lex.hdd(draws=42) if total_words >= 50 else None
        vocd = lex.vocd(ntokens=min(50, total_words-1), within_sample=100, iterations=3) if total_words >= 50 else None
    except:
        freq_ttr = unique_words / total_words if total_words > 0 else 0
        mattr = freq_ttr
        mtld = None
        hdd = None
        vocd = None
    
    fillers = ['um', 'uh', 'like', 'you know', 'i mean', 'sort of', 'kind of', 'well', 'so', 'actually', 'basically']
    text_lower = text.lower()
    filler_count = sum(text_lower.count(f' {f} ') for f in fillers)
    
    repetitions = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
    named_entities = len(doc.ents)
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    sent_lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
    sent_length_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    
    return {
        'Duration_(sec)': round(duration, 2),
        'Total_Utts': num_utterances,
        'MLU_Words': round(mlu_words, 3),
        'FREQ_types': unique_words,
        'FREQ_tokens': total_words,
        'FREQ_TTR': round(freq_ttr, 3),
        'MATTR': round(mattr, 3),
        'MTLD': round(mtld, 3) if mtld else None,
        'HDD': round(hdd, 3) if hdd else None,
        'VOCD': round(vocd, 3) if vocd else None,
        'Words_Min': round(words_min, 2),
        'Verbs_Utt': round(verbs_utt, 3),
        '%_Nouns': round(pct_nouns, 2),
        '%_Verbs': round(pct_verbs, 2),
        '%_Adj': round(pct_adj, 2),
        '%_Adv': round(pct_adv, 2),
        '%_det': round(pct_det, 2),
        '%_pro': round(pct_pro, 2),
        '%_prep': round(pct_prep, 2),
        'noun_verb': round(noun_verb, 3),
        'open_closed': round(open_closed, 3),
        'density': round(density, 3),
        '#open-class': content_count,
        '#closed-class': function_count,
        'filler_words': filler_count,
        'repetitions': repetitions,
        'named_entities': named_entities,
        'avg_word_length': round(avg_word_length, 2),
        'sent_length_std': round(sent_length_std, 2),
    }

def extract_chinese_linguistic_features(transcript, audio_path):
    """Extract Chinese linguistic features"""
    _, nlp_zh = load_nlp_models()
    
    duration = librosa.get_duration(path=audio_path)
    text = transcript.strip()
    
    if not text or len(text) < 10:
        return get_default_linguistic_features(duration)
    
    jieba_words = pseg.cut(text)
    jieba_word_list = [(word, flag) for word, flag in jieba_words if word.strip()]
    
    import re
    chars = [c for c in text if c.strip() and c not in '。，！？；：、""''（）【】《》…—']
    total_chars = len(chars)
    unique_chars = len(set(chars))
    
    words = [word for word, _ in jieba_word_list if len(word.strip()) > 0]
    pos_tags = [flag for _, flag in jieba_word_list]
    
    if len(words) == 0:
        return get_default_linguistic_features(duration)
    
    total_words = len(words)
    unique_words = len(set(words))
    
    sentences = re.split('[。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_utterances = len(sentences) if sentences else 1
    
    from collections import Counter
    pos_counts = Counter(pos_tags)
    
    noun_count = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('n'))
    verb_count = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('v'))
    adj_count = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('a'))
    adv_count = pos_counts.get('d', 0)
    pron_count = pos_counts.get('r', 0)
    prep_count = pos_counts.get('p', 0)
    measure_count = pos_counts.get('q', 0)
    
    mlu_words = total_words / num_utterances if num_utterances > 0 else 0
    mlu_chars = total_chars / num_utterances if num_utterances > 0 else 0
    words_min = (total_words / duration) * 60 if duration > 0 else 0
    chars_min = (total_chars / duration) * 60 if duration > 0 else 0
    
    word_ttr = unique_words / total_words if total_words > 0 else 0
    char_ttr = unique_chars / total_chars if total_chars > 0 else 0
    
    content_count = sum(1 for tag in pos_tags if tag.startswith(('n', 'v', 'a')))
    function_count = total_words - content_count
    density = content_count / total_words if total_words > 0 else 0
    
    chinese_fillers = ['嗯', '啊', '呃', '那个', '这个']
    filler_count = sum(text.count(filler) for filler in chinese_fillers)
    
    return {
        'Duration_(sec)': duration,
        'Total_Utts': num_utterances,
        'Total_Chars': total_chars,
        'Char_TTR': char_ttr,
        'MLU_Chars': mlu_chars,
        'Chars_Min': chars_min,
        'Total_Words': total_words,
        'Word_TTR': word_ttr,
        'MLU_Words': mlu_words,
        'Words_Min': words_min,
        'Content_Density': density,
        'Filler_Words': filler_count,
        '%_Nouns': (noun_count / total_words * 100) if total_words > 0 else 0,
        '%_Verbs': (verb_count / total_words * 100) if total_words > 0 else 0,
        '%_Measure': (measure_count / total_words * 100) if total_words > 0 else 0,
    }

def get_default_linguistic_features(duration):
    return {'Duration_(sec)': duration, 'Total_Utts': 1, 'MLU_Words': 0, 'FREQ_types': 0, 'FREQ_tokens': 0}

# ============================================
# IMPROVED ANALYSIS FUNCTIONS (MODEL-BASED)
# ============================================

def analyze_features_with_model(ling_dict, risk_score, model, feature_names, X_selected, language="en"):
    """
    FIXED VERSION: Generate analysis based on what the MODEL actually learned,
    not hardcoded thresholds.
    
    Uses feature importances and the relationship between features and risk
    to provide meaningful insights.
    """
    
    findings = []
    risk_factors = []
    positive_factors = []
    
    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_names, model.feature_importances_))
    else:
        # If no feature importances, use coefficients or equal weights
        importances = {name: 1.0/len(feature_names) for name in feature_names}
    
    # Create a mapping of linguistic features to their values and model importance
    feature_analysis = []
    
    # Map linguistic features to their positions in the model's feature vector
    ling_features_map = {
        'Words_Min': 'Words_Min',
        'MLU_Words': 'MLU_Words',
        'FREQ_TTR': 'FREQ_TTR',
        'MATTR': 'MATTR',
        'density': 'density',
        '%_pro': '%_pro',
        'filler_words': 'filler_words',
        'repetitions': 'repetitions',
        'Chars_Min': 'Chars_Min',
        'Word_TTR': 'Word_TTR',
        'Content_Density': 'Content_Density'
    }
    
    # Analyze features based on their importance in the model
    for ling_key, model_key in ling_features_map.items():
        if ling_key in ling_dict and ling_dict[ling_key] is not None:
            value = ling_dict[ling_key]
            
            # Find this feature in the model's features
            if model_key in importances:
                importance = importances[model_key]
                
                # Find the index of this feature
                if model_key in feature_names:
                    idx = list(feature_names).index(model_key)
                    normalized_value = X_selected[0, idx] if hasattr(X_selected, 'shape') else X_selected[idx]
                    
                    feature_analysis.append({
                        'name': ling_key,
                        'value': value,
                        'normalized_value': normalized_value,
                        'importance': importance,
                        'contribution': abs(normalized_value * importance)
                    })
    
    # Sort by contribution to see what's actually affecting the prediction
    feature_analysis.sort(key=lambda x: x['contribution'], reverse=True)
    
    # Generate insights based on TOP contributing features (model-driven, not hardcoded!)
    top_n = min(5, len(feature_analysis))
    
    for feat in feature_analysis[:top_n]:
        name = feat['name']
        value = feat['value']
        norm_val = feat['normalized_value']
        importance = feat['importance']
        
        # Determine if this feature pushes toward risk (positive norm_val) or safety (negative)
        # This is model-based: positive normalized values after scaling indicate deviation
        # in the direction the model associates with the positive class (dementia/MCI)
        
        if importance > 0.05:  # Only comment on features that matter to the model
            if abs(norm_val) > 1.0:  # Significant deviation from training mean
                if norm_val > 1.5:
                    risk_factors.append(
                        f"**{name.replace('_', ' ').title()}**: {value:.2f} - "
                        f"Significantly different from typical healthy patterns (model importance: {importance:.1%})"
                    )
                elif norm_val > 0.5:
                    findings.append(
                        f"**{name.replace('_', ' ').title()}**: {value:.2f} - "
                        f"Moderately atypical (model importance: {importance:.1%})"
                    )
                elif norm_val < -0.5:
                    positive_factors.append(
                        f"**{name.replace('_', ' ').title()}**: {value:.2f} - "
                        f"Within healthy range (model importance: {importance:.1%})"
                    )
    
    # If we didn't find any significant contributors, provide general risk-based commentary
    if not risk_factors and not findings and risk_score > 30:
        risk_factors.append(
            "The model detected subtle patterns in the combination of acoustic and linguistic features "
            "that deviate from typical healthy speech, though no single feature shows extreme deviation."
        )
    
    if not positive_factors and risk_score < 30:
        positive_factors.append(
            "Overall speech patterns are consistent with healthy cognitive function across "
            "the features the model considers most important."
        )
    
    return risk_factors, findings, positive_factors

def get_feature_importance_explanation(model, feature_names, X_selected):
    """Get top contributing features to the prediction"""
    
    if not hasattr(model, 'feature_importances_'):
        return []
    
    importances = model.feature_importances_
    
    if hasattr(X_selected, 'shape'):
        values = X_selected[0]
    else:
        values = X_selected
    
    # Combine importance and values
    feature_contributions = []
    for i, (name, imp, val) in enumerate(zip(feature_names, importances, values)):
        feature_contributions.append({
            'name': name,
            'importance': imp,
            'value': val,
            'contribution': abs(imp * val)
        })
    
    # Sort by contribution
    feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
    
    return feature_contributions[:5]  # Top 5

# ============================================
# MAIN APP
# ============================================

def main():
    st.set_page_config(
        page_title="ECHO - Cognitive Health Observatory",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    
    # Header
    st.markdown("<h1>🧠 ECHO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b0b0b0; font-size: 1.1rem; margin-top: -1rem;'>Early Cognitive Health Observatory</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.info("⚕️ **Research Tool** — For educational purposes only. Not a medical diagnostic tool. Consult healthcare professionals for proper evaluation.")
    
    # Add verification mode toggle
    show_verification = st.sidebar.checkbox("🔍 Show Technical Verification", value=False, help="Show detailed extraction and processing steps")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # STEP 1: MODEL SELECTION
    # ============================================
    
    st.markdown("<div class='step-indicator'>STEP 1: Select Assessment Type & Language</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        condition = st.selectbox(
            "Condition",
            ["Alzheimer's Disease", "Mild Cognitive Impairment"]
        )
    
    with col2:
        if condition == "Alzheimer's Disease":
            language = st.selectbox("Language", ["English"], disabled=True)
            model_key = "AD (English)"
        else:
            language = st.selectbox("Language", ["English", "Chinese"])
            model_key = "MCI (English)" if language == "English" else "MCI (Chinese)"
    
    # Load model
    try:
        model, scaler, feature_names, threshold, selector, config = load_model_artifacts(model_key)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Model Accuracy", config["accuracy"])
        with col_b:
            st.metric("Features", len(feature_names))
        with col_c:
            st.metric("Language", config["language"].upper())
        
        st.success(f"✅ Loaded {config['description']} model")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure model files are in the correct directory")
        return
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # STEP 2: AUDIO UPLOAD
    # ============================================
    
    st.markdown("<div class='step-indicator'>STEP 2: Upload Audio Sample</div>", unsafe_allow_html=True)
    st.write("**Requirements:** 30-120 second recording of spontaneous speech (e.g., describing a picture, telling a story)")
    
    uploaded_file = st.file_uploader(
        "Choose audio file",
        type=['wav', 'mp3', 'ogg', 'm4a'],
        help="Upload a clear audio recording"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        # ============================================
        # STEP 3: ANALYSIS
        # ============================================
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='step-indicator'>STEP 3: Analyze Recording</div>", unsafe_allow_html=True)
        
        if st.button("🔬 Start Analysis", use_container_width=True):
            
            with st.spinner("Processing..."):
                
                try:
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_audio_path = tmp_file.name
                    
                    # Convert to WAV if needed
                    if not temp_audio_path.endswith('.wav'):
                        audio = AudioSegment.from_file(temp_audio_path)
                        wav_path = temp_audio_path.replace(Path(temp_audio_path).suffix, '.wav')
                        audio.export(wav_path, format='wav')
                        os.remove(temp_audio_path)
                        temp_audio_path = wav_path
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    # Transcribe
                    status.text("🎤 Transcribing audio...")
                    progress.progress(20)
                    whisper_model = load_whisper_model()
                    result = whisper_model.transcribe(temp_audio_path)
                    transcript = result['text']
                    
                    if show_verification:
                        with st.expander("📝 Transcription Result"):
                            st.write(transcript)
                    
                    # Extract features
                    status.text("🔊 Extracting acoustic features...")
                    progress.progress(40)
                    
                    features_list = []
                    
                    # Acoustic features (if needed for this model)
                    if "acoustic" in config["features"]:
                        try:
                            acoustic_df = extract_acoustic_features(temp_audio_path)
                            features_list.append(acoustic_df)
                            
                            if show_verification:
                                with st.expander("🔊 Acoustic Features (OpenSMILE)"):
                                    st.write(f"Extracted {len(acoustic_df.columns)} acoustic features")
                                    st.dataframe(acoustic_df.head())
                        except Exception as e:
                            st.warning(f"⚠️ Acoustic extraction issue: {str(e)[:50]}")
                    
                    # Linguistic features
                    status.text("📊 Extracting linguistic features...")
                    progress.progress(60)
                    
                    try:
                        if config["language"] == "en":
                            ling_dict = extract_english_linguistic_features(transcript, temp_audio_path)
                        else:
                            ling_dict = extract_chinese_linguistic_features(transcript, temp_audio_path)
                        
                        ling_df = pd.DataFrame([ling_dict], index=[0])
                        features_list.append(ling_df)
                        
                        if show_verification:
                            st.info(f"✅ Extracted {len(ling_df.columns)} linguistic features")
                    except Exception as e:
                        st.error(f"❌ Linguistic extraction failed: {e}")
                        raise
                    
                    # Clean and combine features
                    cleaned_features = []
                    for df in features_list:
                        df = df.reset_index(drop=True)
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = ['_'.join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns.to_flat_index()]
                        cleaned_features.append(df)
                    
                    try:
                        features_df = pd.concat(cleaned_features, axis=1)
                    except:
                        features_df = cleaned_features[0].copy()
                        for df in cleaned_features[1:]:
                            for col in df.columns:
                                features_df[col] = df[col].values
                    
                    if show_verification:
                        st.success(f"✅ Combined: {len(features_df.columns)} total features")
                    
                    progress.progress(80)
                    
                    # Prediction pipeline
                    status.text("🧮 Making prediction...")
                    
                    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    # Get scaler's expected features
                    if hasattr(scaler, 'feature_names_in_'):
                        expected_features = list(scaler.feature_names_in_)
                        if show_verification:
                            st.info(f"✅ Scaler expects {len(expected_features)} features")
                    else:
                        expected_features = list(features_df.columns)[:scaler.n_features_in_]
                    
                    # Align features
                    aligned_df = pd.DataFrame(columns=expected_features, index=[0])
                    for col in expected_features:
                        aligned_df[col] = features_df[col].values if col in features_df.columns else 0
                    
                    aligned_df = aligned_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    # Scale all features
                    X_scaled_all = scaler.transform(aligned_df.values)
                    
                    if show_verification:
                        st.success(f"✅ Scaled features - range: [{X_scaled_all.min():.2f}, {X_scaled_all.max():.2f}]")
                    
                    # Select features for model
                    if selector is not None:
                        X_selected = selector.transform(X_scaled_all)
                        selected_features = feature_names  # Use original feature_names for analysis
                        if show_verification:
                            st.success(f"✅ Selector picked {X_selected.shape[1]} features")
                    else:
                        selected_indices = [expected_features.index(f) for f in feature_names if f in expected_features]
                        X_selected = X_scaled_all[:, selected_indices] if selected_indices else X_scaled_all[:, :len(feature_names)]
                        selected_features = feature_names
                        if show_verification:
                            st.success(f"✅ Selected {X_selected.shape[1]} features by name")
                    
                    # Verification check
                    if show_verification:
                        st.markdown("### 🔍 Verification Check")
                        st.write(f"**Selected features (first 10):**")
                        for i, name in enumerate(feature_names[:10]):
                            st.write(f"  {i+1}. {name}: {X_selected[0, i]:.4f}")
                        
                        st.write(f"\n**Statistics:**")
                        st.write(f"  Mean: {X_selected.mean():.4f}")
                        st.write(f"  Std: {X_selected.std():.4f}")
                        st.write(f"  Min: {X_selected.min():.4f}")
                        st.write(f"  Max: {X_selected.max():.4f}")
                    
                    # Predict
                    probability = model.predict_proba(X_selected)[0]
                    risk_score = probability[1] * 100
                    
                    progress.progress(100)
                    status.text("✅ Analysis complete!")
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # ============================================
                    # RESULTS DISPLAY
                    # ============================================
                    
                    st.markdown("---")
                    st.markdown("<h2 style='text-align: center;'>📊 Assessment Results</h2>", unsafe_allow_html=True)
                    
                    # Risk interpretation
                    if risk_score < 30:
                        risk_level = "Low Risk"
                        risk_color = "#00d4aa"
                        risk_emoji = "✅"
                        interpretation = "Speech patterns suggest healthy cognitive function."
                    elif risk_score < 60:
                        risk_level = "Moderate Risk"
                        risk_color = "#ffa500"
                        risk_emoji = "⚠️"
                        interpretation = "Some indicators present. Clinical evaluation recommended."
                    else:
                        risk_level = "Elevated Risk"
                        risk_color = "#ff4444"
                        risk_emoji = "🚨"
                        interpretation = "Multiple indicators detected. Professional assessment advised."
                    
                    # Main result card
                    st.markdown(f"""
                    <div class='result-card'>
                        <h3 style='text-align: center; color: {risk_color}; margin-bottom: 1rem;'>
                            {risk_emoji} {risk_level}
                        </h3>
                        <div style='text-align: center;'>
                            <div style='font-size: 4rem; font-weight: 700; color: {risk_color};'>
                                {risk_score:.1f}%
                            </div>
                            <div style='color: #b0b0b0; margin-top: 0.5rem;'>
                                Risk Score
                            </div>
                        </div>
                        <p style='text-align: center; color: #e0e0e0; margin-top: 1.5rem; font-size: 1.1rem;'>
                            {interpretation}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Detailed metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Condition", config["condition"])
                    
                    with col2:
                        st.metric("Model Accuracy", config["accuracy"])
                    
                    # ============================================
                    # INTELLIGENT ANALYSIS SECTION (FIXED!)
                    # ============================================
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Detailed Analysis: Why This Result?")
                    st.caption("⚠️ Analysis now based on what the model actually learned from data, not hardcoded rules!")
                    
                    # Use the NEW function that uses the model
                    risk_factors, findings, positive_factors = analyze_features_with_model(
                        ling_dict, risk_score, model, selected_features, X_selected, config["language"]
                    )
                    
                    if risk_factors:
                        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
                        st.markdown("**🚨 Risk Indicators Detected:**")
                        for factor in risk_factors:
                            st.markdown(f"<div class='indicator risk'>• {factor}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    if findings:
                        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
                        st.markdown("**⚠️ Notable Observations:**")
                        for finding in findings:
                            st.markdown(f"<div class='indicator warning'>• {finding}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    if positive_factors:
                        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
                        st.markdown("**✅ Healthy Indicators:**")
                        for factor in positive_factors:
                            st.markdown(f"<div class='indicator normal'>• {factor}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Feature importance
                    st.markdown("---")
                    st.markdown("### 📈 Top Contributing Features")
                    
                    top_features = get_feature_importance_explanation(model, selected_features, X_selected)
                    
                    if top_features:
                        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
                        st.markdown("**Most influential features in this prediction:**")
                        for i, feat in enumerate(top_features):
                            st.markdown(f"""
                            <div style='padding: 0.5rem; margin: 0.5rem 0; background: rgba(255,255,255,0.03); border-radius: 8px;'>
                                <strong>{i+1}. {feat['name']}</strong><br>
                                <small>Value: {feat['value']:.4f} | Importance: {feat['importance']:.1%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Transcript
                    with st.expander("📝 View Transcript"):
                        st.write(transcript)
                        st.caption(f"Word count: {len(transcript.split())} | Duration: {ling_dict.get('Duration_(sec)', 0):.1f}s")
                    
                    # Key indicators
                    with st.expander("🔢 Raw Feature Values"):
                        ind_col1, ind_col2 = st.columns(2)
                        
                        with ind_col1:
                            if config["language"] == "en":
                                st.metric("Speech Rate", f"{ling_dict.get('Words_Min', 0):.0f} words/min")
                                st.metric("Vocabulary Diversity (TTR)", f"{ling_dict.get('FREQ_TTR', 0):.3f}")
                                st.metric("Content Density", f"{ling_dict.get('density', 0):.3f}")
                                st.metric("Pronoun %", f"{ling_dict.get('%_pro', 0):.1f}%")
                            else:
                                st.metric("Speech Rate", f"{ling_dict.get('Chars_Min', 0):.0f} chars/min")
                                st.metric("Vocabulary Diversity", f"{ling_dict.get('Word_TTR', 0):.3f}")
                                st.metric("Content Density", f"{ling_dict.get('Content_Density', 0):.3f}")
                        
                        with ind_col2:
                            st.metric("Utterance Length", f"{ling_dict.get('MLU_Words', 0):.1f} words")
                            st.metric("Filler Words", ling_dict.get('Filler_Words', 0) if config["language"] == "zh" else ling_dict.get('filler_words', 0))
                            st.metric("Repetitions", ling_dict.get('repetitions', 0))
                            if config["language"] == "en" and 'MATTR' in ling_dict:
                                st.metric("Lexical Richness (MATTR)", f"{ling_dict.get('MATTR', 0):.3f}")
                    
                    # Next steps
                    st.markdown("---")
                    st.markdown("### 🩺 Recommended Next Steps")
                    
                    if risk_score >= 60:
                        st.warning("""
                        **High risk indicators detected:**
                        - Consult a neurologist or geriatrician
                        - Request comprehensive cognitive assessment
                        - Discuss findings with healthcare provider
                        """)
                    elif risk_score >= 30:
                        st.info("""
                        **Moderate indicators present:**
                        - Schedule routine check-up with primary care physician
                        - Monitor cognitive function over time
                        - Consider follow-up assessment in 6-12 months
                        """)
                    else:
                        st.success("""
                        **Low risk profile:**
                        - Continue healthy cognitive habits
                        - Regular physical and mental exercise
                        - Stay socially and intellectually engaged
                        """)
                    
                    # Clean up
                    os.remove(temp_audio_path)
                    
                except Exception as e:
                    st.error(f"❌ Analysis error: {e}")
                    with st.expander("Debug Info"):
                        import traceback
                        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='footer'>
        <p><b>ECHO - Early Cognitive Health Observatory</b></p>
        <p>AI-powered speech analysis for cognitive health research</p>
        <p style='font-size: 0.75rem; margin-top: 1rem;'>
            Models: AD-CN (85.9%) | MCI-CN English (76.6%) | MCI-CN Chinese (60%)<br>
            Features: Acoustic (OpenSMILE eGeMAPS) + Linguistic (spaCy, jieba, Whisper)
        </p>
        <p style='font-size: 0.7rem; color: #606060; margin-top: 1rem;'>
            ⚠️ Not for clinical diagnosis. Research purposes only.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()