import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# === CONFIGURATION ===
folder_path = r"C:\Users\orenh\Downloads"
files = ["data.mat", "data (1).mat", "data (2).mat", "data (3).mat", "data (4).mat", "data (5).mat"]
dates = ["01_22_19", "01_24_19", "01_28_19", "02_03_19", "02_05_19", "02_24_19"]
animal_id = "4458"

window_size = 30
step_size = 15
n_splits = 5

# Storage for all results
results = []

def compute_chance(y):
    s = np.sum(y == 1)
    f = np.sum(y == 0)
    return max(s / (s + f), 1 - s / (s + f)), s, f

# Loop over all files and dates
for file_name, date in zip(files, dates):
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping.")
        continue

    mat = loadmat(file_path, simplify_cells=True)

    if 'imagingData' not in mat or 'samples' not in mat['imagingData']:
        print(f"Invalid structure in {file_name}, skipping.")
        continue

    data = mat['imagingData']['samples']  # [neurons, time, trials]
    labels = np.array(mat['BehaveData']['success']['indicatorPerTrial']).astype(int).flatten()
    roi_names = [f"ROI_{i}" for i in range(data.shape[0])]

    if len(labels) != data.shape[2]:
        print(f"Mismatch between labels and trials in {file_name}, skipping.")
        continue

    chance, s, f = compute_chance(labels)
    windows = [(start, start + window_size) for start in range(0, data.shape[1] - window_size + 1, step_size)]

    for i, roi in enumerate(roi_names):
        acc_per_window = []
        success_values = []
        failure_values = []
        windows_above_chance = 0

        for start, end in windows:
            activity = np.nanmean(data[i, start:end, :], axis=0)
            X = activity.reshape(-1, 1)
            y = labels

            if np.isnan(X).any() or len(np.unique(y)) < 2:
                continue

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_accuracies = []

            for train_idx, test_idx in skf.split(X, y):
                model = LogisticRegression(solver='liblinear')
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                fold_accuracies.append(accuracy_score(y[test_idx], preds))

            mean_acc = np.mean(fold_accuracies)
            acc_per_window.append(mean_acc)

            if mean_acc > chance:
                windows_above_chance += 1

            success_values.extend(X[y == 1].flatten())
            failure_values.extend(X[y == 0].flatten())

        final_acc = np.mean(acc_per_window) if acc_per_window else np.nan

        if len(success_values) > 1 and len(failure_values) > 1:
            mean_diff = np.mean(success_values) - np.mean(failure_values)
            pooled_std = np.sqrt(0.5 * (np.var(success_values) + np.var(failure_values)))
            dprime = mean_diff / pooled_std if pooled_std > 0 else 0
        else:
            dprime = 0

        results.append({
            "Animal": animal_id,
            "Date": date,
            "ROI": roi,
            "Success Trials": s,
            "Failure Trials": f,
            "Chance Level": round(chance, 3),
            "Mean Accuracy (CV)": round(final_acc, 3),
            "# Windows > Chance": windows_above_chance,
            "Sensitivity Index (d′)": round(dprime, 3)
        })

# Save everything to Excel
df = pd.DataFrame(results)

output_path = os.path.join(folder_path, f"results_{animal_id}_all_flavor_dates.xlsx")
df.to_excel(output_path, index=False)

print("\nResults saved successfully to:")
print(output_path)
