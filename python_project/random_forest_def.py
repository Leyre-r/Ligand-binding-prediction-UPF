"""
random_forest_def.py
------------------------
Random Forest using SAS-based structural descriptors derived from PDB structures. 

Use:
    python3 random_forest_def.py dataset_test.csv

Output:
    modelo_rf_predictor.pkl  → serialized (.joblib) Random Forest model
"""

# Data processing
import pandas as pd
import numpy as np

# Data Modelling
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
import joblib
import sys

# ─────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────
if len(sys.argv) < 2:
    print("Use: python random_forest_def.py dataset_test.csv")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

X = df.drop(columns=['target', 'pdb_id'])
y = df['target']
groups = df['pdb_id']

# ─────────────────────────────
# 2. SPLIT POR PROTEÍNA
# ─────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]

print("Proteínas en train:", len(set(groups_train)))
print("Proteínas en test:", len(set(groups.iloc[test_idx])))

# comprobar overlap
overlap = set(groups_train) & set(groups.iloc[test_idx])
print("Overlap:", len(overlap))

# ─────────────────────────────
# 3. MODELO BASE
# ─────────────────────────────
rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=2
)

# ─────────────────────────────
# 4. GRID SEARCH (SIN LEAKAGE)
# ─────────────────────────────
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

gkf = GroupKFold(n_splits=3)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=gkf.split(X_train, y_train, groups=groups_train),
    scoring='roc_auc',
    verbose=2,
    n_jobs=2
)

print("\n🔍 Buscando mejores hiperparámetros...")
grid_search.fit(X_train, y_train)

mejor_rf = grid_search.best_estimator_

print("\nMejores parámetros:", grid_search.best_params_)

# ─────────────────────────────
# 8. GUARDAR MODELO
# ─────────────────────────────

joblib.dump(mejor_rf, 'modelo_rf_predictor.pkl') #Gemin i propone cambiar el formato a .joblib
print("\nModelo guardado como modelo_rf_predictor.pkl")

# ─────────────────────────────
# 5. EVALUACIÓN FINAL
# ─────────────────────────────
y_proba = mejor_rf.predict_proba(X_test)[:, 1]

# threshold ajustable
threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# ─────────────────────────────
# 6. MATRIZ DE CONFUSIÓN
# ─────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Binding', 'Binding'],
            yticklabels=['No Binding', 'Binding'])
plt.title('Matriz de Confusión — Modelo RF')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ─────────────────────────────
# 7. IMPORTANCIA DE FEATURES
# ─────────────────────────────
importancias = pd.Series(
    mejor_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n--- Importancia de las Features ---")
print(importancias.head(20))

#gráfico de importancias
plt.figure(figsize=(10, 6))
importancias.head(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png')
plt.show()

