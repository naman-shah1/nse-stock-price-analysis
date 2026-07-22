import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("final_test_predictions_2025H2.csv")

thresholds = np.linspace(-0.1, 0.1, 41)
results = []

true_vals = df['true_y_t1'].values
pred_vals = df['pred_y_t1'].values
y_true_up = (true_vals > 0).astype(int)

for t in thresholds:
    y_pred_up = (pred_vals > t).astype(int)
    
    tp = np.sum((y_true_up == 1) & (y_pred_up == 1))
    fp = np.sum((y_true_up == 0) & (y_pred_up == 1))
    fn = np.sum((y_true_up == 1) & (y_pred_up == 0))
    tn = np.sum((y_true_up == 0) & (y_pred_up == 0))
    
    results.append({
        'threshold': t,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })

res_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))

# Plot FN and FP
plt.plot(res_df['threshold'], res_df['fn'], label='False Negatives (Missed UP)', color='red', linewidth=2)
plt.plot(res_df['threshold'], res_df['fp'], label='False Positives (Bad BUY)', color='orange', linewidth=2)

# Plot TP
plt.plot(res_df['threshold'], res_df['tp'], label='True Positives (Good BUY)', color='green', linewidth=2)

plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Current Threshold (0.0)')

plt.title('Tradeoff Analysis: FN / FP / TP vs Decision Threshold (Horizon 1)')
plt.xlabel('Decision Threshold (Predicted Scaled Return)')
plt.ylabel('Number of Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\ARIHANT IMPEX\.gemini\antigravity\brain\ae159755-5310-4add-ad4d-def7ee5158b3\threshold_analysis.png', dpi=150)
print("Plot saved.")
