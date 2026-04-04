import matplotlib.pyplot as plt

languages = ['English AD', 'Spanish AD', 'Chinese AD', 'Chinese MCI']
aucs = [0.879, 0.860, 0.779, 0.586]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

plt.figure(figsize=(10, 6))
bars = plt.barh(languages, aucs, color=colors)
plt.xlabel('ROC-AUC Score', fontsize=12)
plt.title('Cross-Linguistic Performance Comparison', fontsize=14, fontweight='bold')
plt.xlim([0, 1.0])
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

for i, v in enumerate(aucs):
    plt.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figure4_roc_comparison.png', dpi=300, bbox_inches='tight')