from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

model_features = {
    "XGBoost": [
        "oc_Opdr. Kweken afnemen",
        "Noradrenaline (Norepinefrine)_mean_dose",
        "Fentanyl_total_dose",
        "Fentanyl_mean_dose",
        "Calcium Glubionaat (Calcium Sandoz)_mean_dose",
        "Paracetamol_mean_dose",
        "UrineCAD_sd",
        "UrineCAD_mean",
        "Cefazoline (Kefzol)_mean_dose",
        "Saturatie (Monitor)_mean",
    ],
    "LightGBM": [
        "oc_Opdr. Kweken afnemen",
        "Noradrenaline (Norepinefrine)_mean_dose",
        "Fentanyl_mean_dose",
        "Fentanyl_total_dose",
        "Calcium Glubionaat (Calcium Sandoz)_mean_dose",
        "Paracetamol_mean_dose",
        "UrineCAD_sd",
        "oc_Voeding Drinken",
        "Thiamine (Vitamine B1)_mean_dose",
        "Saturatie (Monitor)_mean",
    ],
    "LogisticRegression": [
        "process_Beademen",
        "process_Tube",
        "process_Trilumen Jugularis",
        "Noradrenaline (Norepinefrine)_mean_dose",
        "oc_Opdr. Kweken afnemen",
        "Calcium Glubionaat (Calcium Sandoz)_mean_dose",
        "O2-Saturatie (bloed)_sd",
        "process_Sonde",
        "Calcium Glubionaat (Calcium Sandoz)_total_dose",
        "oc_2. Spuitpompen",
    ],
    "CatBoost": [
        "specialty",
        "oc_Opdr. Kweken afnemen",
        "Fentanyl_mean_dose",
        "Noradrenaline (Norepinefrine)_mean_dose",
        "Noradrenaline (Norepinefrine)_total_dose",
        "Calcium Glubionaat (Calcium Sandoz)_mean_dose",
        "Paracetamol_total_dose",
        "Fentanyl_total_dose",
        "Saturatie (Monitor)_mean",
        "process_Trilumen Jugularis",
    ],
    "MLP": [
        "Gelofusine_total_dose",
        "Paracetamol_total_dose",
        "Paracetamol_mean_dose",
        "Gelofusine_mean_dose",
        "Propofol (Diprivan)_total_dose",
        "Ri-Lac (Ringers lactaat)_total_dose",
        "Thiamine (Vitamine B1)_mean_dose",
        "ABP diastolisch_mean",
        "Ri-Lac (Ringers lactaat)_mean_dose",
        "NaCl 0,9 %_total_dose",
    ],
    "DecisionTree": [
        "Fentanyl_mean_dose",
        "Calcium Glubionaat (Calcium Sandoz)_total_dose",
        "Act.HCO3 (bloed)_sd",
        "B.E. (bloed)_mean",
        "oc_Opdr. Circulatie",
        "Noradrenaline (Norepinefrine)_mean_dose",
        "process_Trilumen Jugularis",
        "Noradrenaline (Norepinefrine)_total_dose",
        "specialty",
        "oc_Opdr. Kweken afnemen",
    ],
    "RandomForest": [
        "Fentanyl_mean_dose",
        "Fentanyl_total_dose",
        "Noradrenaline (Norepinefrine)_mean_dose",
        "oc_Opdr. Kweken afnemen",
        "Calcium Glubionaat (Calcium Sandoz)_mean_dose",
        "Noradrenaline (Norepinefrine)_total_dose",
        "Calcium Glubionaat (Calcium Sandoz)_total_dose",
        "Cefazoline (Kefzol)_total_dose",
        "specialty",
        "UrineCAD_sd",
    ],
    "TMLP": [
        "NaCL 0,9% spuit_total_dose",
        "oc_Opdr. Circulatie",
        "NaCL 0,9% spuit_mean_dose",
        "Propofol (Diprivan)_mean_dose",
        "Gefiltreerde Ery's_total_dose",
        "Paracetamol_mean_dose",
        "oc_Opdr. Overig",
        "Glucose (bloed)_mean",
        "SDD drank (4 x dgs)_mean_dose",
        "NaCl 0,45%/Glucose 2,5%_mean_dose",
    ],
}

# Count top-10 appearances
top10_counter = Counter()
for feats in model_features.values():
    top10_counter.update(feats)  # all 10 features


models_per_feature_top10 = defaultdict(list)

for model, feats in model_features.items():
    for f in feats:
        models_per_feature_top10[f].append(model)

for feat, cnt in top10_counter.most_common():
    print(f"{feat:<50} appears in {cnt} models: {models_per_feature_top10[feat]}")

# Count feature frequencies and filter 
feature_counter = Counter()
for feats in model_features.values():
    feature_counter.update(feats)

# Sort descending so most common first
df_counts = (
    pd.DataFrame.from_dict(feature_counter, orient='index', columns=['count'])
      .query('count >= 3')
      .sort_values('count', ascending=False)
)

features = df_counts.index.tolist()

models = [
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "LogisticRegression",
    "MLP",
    "TMLP"
]
presence = pd.DataFrame(0, index=features, columns=models)
for model, feats in model_features.items():
    for f in feats:
        if f in features and model in models:
            presence.at[f, model] = 1

plt.figure(figsize=(10, 6))
cmap = ListedColormap(['#f7f7f7', '#3182bd'])
plt.imshow(presence, aspect='auto', cmap=cmap, origin='upper')
plt.xticks(range(len(models)), models, rotation=45, ha='right')
plt.yticks(range(len(features)), features)
plt.xlabel("Models")
plt.ylabel("SHAP Features")
plt.title("Presence of Key SHAP Features Across Models", fontsize=14)

# Annotate with checkmarks
for i in range(len(features)):
    for j in range(len(models)):
        if presence.iat[i, j]:
            plt.text(j, i, '✓', ha='center', va='center', color='white', fontsize=12)

plt.tight_layout()
plt.savefig("feature_presence_heatmap_custom_order.png", dpi=300)
plt.close()

