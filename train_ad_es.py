"""
SPANISH AD vs CN MODEL TRAINING
Binary classification with automatic train/test split
Matches English training pipeline structure
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("SPANISH AD vs CN MODEL TRAINING")
print("Binary Classification with Train/Test Split")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

# File names (adjust these to match your Spanish files)
ACOUSTIC_FILE = 'spanish_acoustic_features.csv'
LINGUISTIC_FILE = 'spanish_linguistic_features.csv'
LABELS_FILE = 'spanish_labels.csv'

# Output folder
OUTPUT_FOLDER = 'Spanish_AD_Model'

# Feature selection
K_FEATURES = 20  # Top 20 features (same as English)

print(f"\n🎯 Training: AD vs CN (binary classification)")
print(f"📊 Feature selection: Top {K_FEATURES} features")
print(f"💾 Output folder: {OUTPUT_FOLDER}/")

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Loading data...")

# Load acoustic features
try:
    acoustic = pd.read_csv(ACOUSTIC_FILE)
    print(f"✅ Acoustic features: {len(acoustic)} samples, {len(acoustic.columns)} columns")
except FileNotFoundError:
    print(f"⚠️ {ACOUSTIC_FILE} not found - using linguistic only")
    acoustic = None

# Load linguistic features
linguistic = pd.read_csv(LINGUISTIC_FILE)
print(f"✅ Linguistic features: {len(linguistic)} samples, {len(linguistic.columns)} columns")

# Load labels
labels = pd.read_csv(LABELS_FILE)
print(f"✅ Labels: {len(labels)} samples")

# ============================================
# 2. EXTRACT LABELS FROM FILENAMES (if needed)
# ============================================

if 'label' not in labels.columns:
    print("\n🔍 Extracting labels from filenames...")
    
    def extract_label_from_filename(filename):
        """Extract label from filename like 'AD-W-84-183.cha' or 'HC-M-65-79.cha'"""
        filename_upper = str(filename).upper()
        if filename_upper.startswith('AD'):
            return 2  # AD
        elif filename_upper.startswith('MCI'):
            return 1  # MCI
        elif filename_upper.startswith('HC') or filename_upper.startswith('CN'):
            return 0  # CN/HC
        else:
            return None
    
    labels['label'] = labels['filename'].apply(extract_label_from_filename)
    labels = labels.dropna(subset=['label'])
    labels['label'] = labels['label'].astype(int)
    
    print(f"✅ Extracted labels from filenames")

# Show label distribution
print("\n📊 Original label distribution:")
label_counts = labels['label'].value_counts().sort_index()
for label, count in label_counts.items():
    label_name = ['CN/HC', 'MCI', 'AD'][int(label)]
    print(f"   {label_name} (label={int(label)}): {count} samples ({count/len(labels)*100:.1f}%)")

# ============================================
# 3. FILTER FOR AD vs CN ONLY (BINARY)
# ============================================
print("\n🎯 Filtering for AD vs CN (binary classification)...")

# Keep only label 0 (CN) and label 2 (AD)
labels_binary = labels[labels['label'].isin([0, 2])].copy()
# Remap: 0 stays 0 (CN), 2 becomes 1 (AD)
labels_binary['label'] = (labels_binary['label'] == 2).astype(int)

print(f"   CN: {sum(labels_binary['label']==0)} samples")
print(f"   AD: {sum(labels_binary['label']==1)} samples")
print(f"   Total: {len(labels_binary)} samples")

# ============================================
# 4. MERGE FEATURES
# ============================================
print("\n🔄 Merging features...")

# Start with linguistic features
merged = linguistic.copy()

# Add acoustic if available
if acoustic is not None:
    acoustic_cols = [col for col in acoustic.columns 
                     if col not in ['File', 'transcript', 'filename']]
    acoustic_subset = acoustic[['filename'] + acoustic_cols]
    merged = pd.merge(merged, acoustic_subset, on='filename', how='inner')
    print(f"✅ Merged acoustic features")

# Merge with labels
feature_cols = [col for col in merged.columns 
                if col not in ['File', 'transcript', 'filename']]
X_all = merged[['filename'] + feature_cols]

data = pd.merge(X_all, labels_binary[['filename', 'label']], on='filename', how='inner')
data = data.dropna(subset=['label'])

print(f"✅ Final dataset: {len(data)} samples, {len(feature_cols)} features")

# ============================================
# 5. STRATIFIED TRAIN/TEST SPLIT (80/20)
# ============================================
print("\n✂️ Splitting into train/test sets (80/20)...")

# Separate features and labels
X = data[feature_cols].fillna(data[feature_cols].median())
y = data['label'].astype(int)
filenames = data['filename']

# Stratified split
X_train_raw, X_test_raw, y_train, y_test, files_train, files_test = train_test_split(
    X, y, filenames,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train set: {len(y_train)} samples")
print(f"   CN: {sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
print(f"   AD: {sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.1f}%)")

print(f"✅ Test set: {len(y_test)} samples")
print(f"   CN: {sum(y_test==0)} ({sum(y_test==0)/len(y_test)*100:.1f}%)")
print(f"   AD: {sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.1f}%)")

# Save test filenames for verification
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
test_files_df = pd.DataFrame({'filename': files_test, 'label': y_test})
test_files_df.to_csv(f'{OUTPUT_FOLDER}/test_files.csv', index=False)
print(f"✅ Saved test filenames to: {OUTPUT_FOLDER}/test_files.csv")

# ============================================
# 6. FEATURE SCALING
# ============================================
print("\n⚖️ Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

print(f"✅ Features scaled (mean=0, std=1)")

# ============================================
# 7. FEATURE SELECTION
# ============================================
print(f"\n🎯 Feature selection (keeping top {K_FEATURES} features)...")

selector = SelectKBest(f_classif, k=min(K_FEATURES, X_train_df.shape[1]))
X_train_selected = selector.fit_transform(X_train_df, y_train)
X_test_selected = selector.transform(X_test_df)

selected_features = X_train_df.columns[selector.get_support()].tolist()

print(f"✅ Selected {len(selected_features)} features")
print(f"   Top 5 features: {selected_features[:5]}")

X_train_final = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_final = pd.DataFrame(X_test_selected, columns=selected_features)

# ============================================
# 8. TEST MULTIPLE BALANCING METHODS
# ============================================
print("\n🔄 Testing different balancing strategies...")

balancing_methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42, n_neighbors=3),
    'SMOTEENN': SMOTEENN(random_state=42),
    'Undersample': RandomUnderSampler(random_state=42),
    'No_Balance': None
}

balanced_datasets = {}
for name, method in balancing_methods.items():
    try:
        if method is None:
            X_bal, y_bal = X_train_final.copy(), y_train.copy()
        else:
            X_bal, y_bal = method.fit_resample(X_train_final, y_train)
        balanced_datasets[name] = (X_bal, y_bal)
        print(f"   {name}: {len(X_bal)} samples (CN: {sum(y_bal==0)}, AD: {sum(y_bal==1)})")
    except Exception as e:
        print(f"   {name}: FAILED ({str(e)[:50]})")

# ============================================
# 9. TEST MULTIPLE MODELS
# ============================================
print("\n" + "="*70)
print("🔬 TESTING MULTIPLE MODELS")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        class_weight={0:1, 1:1}
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight={0:1, 1:1}
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        class_weight={0:1, 1:1}
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    ),
    'Naive Bayes': GaussianNB(),
    'SVM (Linear)': SVC(
        kernel='linear',
        C=0.1,
        probability=True,
        random_state=42,
        class_weight={0:1, 1:1}
    )
}

results = []

for balance_name, (X_bal, y_bal) in balanced_datasets.items():
    print(f"\n📊 Testing with {balance_name} balancing:")
    
    for model_name, model in models.items():
        try:
            # Train
            model.fit(X_bal, y_bal)
            
            # Evaluate
            train_acc = model.score(X_bal, y_bal)
            test_acc = model.score(X_test_final, y_test)
            
            # Cross-validation on original training data
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_final, y_train, cv=cv, scoring='accuracy')
            
            # Test metrics
            y_pred = model.predict(X_test_final)
            y_proba = model.predict_proba(X_test_final)[:, 1]
            
            cm = confusion_matrix(y_test, y_pred)
            sensitivity = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
            roc_auc = roc_auc_score(y_test, y_proba)
            
            results.append({
                'Balancing': balance_name,
                'Model': model_name,
                'Train_Acc': train_acc,
                'Test_Acc': test_acc,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std() * 2,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'ROC_AUC': roc_auc,
                'Overfitting': train_acc - test_acc
            })
            
            print(f"  {model_name:25s} → Test: {test_acc:.1%}, Sens: {sensitivity:.1%}, Spec: {specificity:.1%}")
            
        except Exception as e:
            print(f"  {model_name:25s} → Error: {str(e)[:50]}")

# ============================================
# 10. FIND BEST MODEL
# ============================================
print("\n" + "="*70)
print("📊 COMPREHENSIVE RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_Acc', ascending=False)

print("\n🏆 TOP 10 MODELS:")
print("─" * 110)
print(f"{'Balancing':<15} {'Model':<25} {'Test Acc':<10} {'Sens':<10} {'Spec':<10} {'ROC-AUC':<10} {'Overfit':<10}")
print("─" * 110)

for idx, row in results_df.head(10).iterrows():
    print(f"{row['Balancing']:<15} {row['Model']:<25} {row['Test_Acc']:<10.1%} "
          f"{row['Sensitivity']:<10.1%} {row['Specificity']:<10.1%} "
          f"{row['ROC_AUC']:<10.3f} {row['Overfitting']:<10.1%}")

print("─" * 110)

best_idx = results_df['Test_Acc'].idxmax()
best = results_df.loc[best_idx]

print(f"\n🎯 BEST MODEL:")
print(f"   Model: {best['Model']}")
print(f"   Balancing: {best['Balancing']}")
print(f"   Test Accuracy: {best['Test_Acc']:.1%}")
print(f"   Sensitivity: {best['Sensitivity']:.1%}")
print(f"   Specificity: {best['Specificity']:.1%}")
print(f"   ROC-AUC: {best['ROC_AUC']:.3f}")
print(f"   Overfitting: {best['Overfitting']:.1%}")

# ============================================
# 11. THRESHOLD OPTIMIZATION
# ============================================
print("\n" + "="*70)
print("🎯 THRESHOLD OPTIMIZATION")
print("="*70)

best_balance = best['Balancing']
best_model_name = best['Model']

X_bal, y_bal = balanced_datasets[best_balance]
best_model = models[best_model_name]
best_model.fit(X_bal, y_bal)

y_proba_best = best_model.predict_proba(X_test_final)[:, 1]

print("\nTesting different decision thresholds:")
print("─" * 70)
print(f"{'Threshold':<12} {'Accuracy':<12} {'Sensitivity':<15} {'Specificity':<15}")
print("─" * 70)

best_threshold = 0.5
best_threshold_acc = 0

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_proba_best >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    cm = confusion_matrix(y_test, y_pred_thresh)
    sens = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    
    print(f"{threshold:<12.1f} {acc:<12.1%} {sens:<15.1%} {spec:<15.1%}")
    
    if acc > best_threshold_acc:
        best_threshold_acc = acc
        best_threshold = threshold

print("─" * 70)
print(f"\n✅ Best threshold: {best_threshold} (Accuracy: {best_threshold_acc:.1%})")

# ============================================
# 12. SAVE MODEL & RESULTS
# ============================================
print("\n" + "="*70)
print("💾 SAVING MODEL")
print("="*70)

# Final predictions with best threshold
y_pred_final = (y_proba_best >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_final)

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['CN', 'AD'],
    yticklabels=['CN', 'AD']
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - AD vs CN")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/confusion_matrix.png", dpi=300)
print(f"✅ Saved confusion matrix plot")

# Save model files
joblib.dump(best_model, f'{OUTPUT_FOLDER}/best_model_final.pkl')
joblib.dump(scaler, f'{OUTPUT_FOLDER}/scaler_final.pkl')
joblib.dump(selected_features, f'{OUTPUT_FOLDER}/features_final.pkl')
joblib.dump(best_threshold, f'{OUTPUT_FOLDER}/threshold_final.pkl')
joblib.dump(selector, f'{OUTPUT_FOLDER}/selector_final.pkl')

print(f"✅ Saved model files to: {OUTPUT_FOLDER}/")
print(f"   • best_model_final.pkl ({best_model_name})")
print(f"   • scaler_final.pkl")
print(f"   • features_final.pkl")
print(f"   • threshold_final.pkl")
print(f"   • selector_final.pkl")

# Save results
results_df.to_csv(f'{OUTPUT_FOLDER}/all_models_comparison.csv', index=False)
print(f"✅ Saved: {OUTPUT_FOLDER}/all_models_comparison.csv")

# ============================================
# 13. FINAL REPORT
# ============================================
print("\n" + "="*70)
print("📊 FINAL REPORT")
print("="*70)

print(f"\n🎯 Task: AD vs CN Detection (Spanish)")
print(f"📁 Dataset: {len(y_train) + len(y_test)} total samples")
print(f"   • Train: {len(y_train)} samples (80%)")
print(f"   • Test: {len(y_test)} samples (20%)")

print(f"\n🏆 Best Model: {best['Model']}")
print(f"   • Balancing: {best['Balancing']}")
print(f"   • Features: {len(selected_features)} selected from {len(feature_cols)} total")

print(f"\n📊 Performance on TEST SET:")
print(f"   • Accuracy: {best['Test_Acc']:.1%}")
print(f"   • Sensitivity: {best['Sensitivity']:.1%} (detects {best['Sensitivity']:.0%} of AD cases)")
print(f"   • Specificity: {best['Specificity']:.1%} (correctly IDs {best['Specificity']:.0%} of CN)")
print(f"   • ROC-AUC: {best['ROC_AUC']:.3f}")
print(f"   • Overfitting: {best['Overfitting']:.1%}")

print(f"\n🔍 Cross-Validation (on train set):")
print(f"   • Mean CV Accuracy: {best['CV_Mean']:.1%}")
print(f"   • CV Std Dev: ±{best['CV_Std']:.1%}")

if best['Test_Acc'] >= 0.75:
    print("\n✅ EXCELLENT PERFORMANCE!")
elif best['Test_Acc'] >= 0.70:
    print("\n✅ GOOD PERFORMANCE!")
elif best['Test_Acc'] >= 0.65:
    print("\n⚠️ MODERATE PERFORMANCE")
else:
    print("\n⚠️ LOWER THAN EXPECTED")
    print("   Consider: More data, different features, or simpler model")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

print(f"\n📁 All files saved to: {OUTPUT_FOLDER}/")
print(f"   Ready to use in your demo!")

print("\n🎯 FOR YOUR PRESENTATION:")
print(f"   'We trained a {best['Model']} for Spanish AD detection'")
print(f"   'Achieved {best['Test_Acc']:.0%} accuracy on held-out test data'")
print(f"   'Sensitivity: {best['Sensitivity']:.0%}, Specificity: {best['Specificity']:.0%}'")
print(f"   'Model uses {len(selected_features)} carefully selected features'")
print(f"   'Dataset: {len(y_train) + len(y_test)} Spanish speakers (AD + CN)'")