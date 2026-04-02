#Data procesing
import pandas as pd

#Data Modelling
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score #otras posibles metricas: precision_score, recall_score, ConfusionMatrixDisplay
import joblib
import numpy as np

# 1. CARGAR DATOS
df = pd.read_csv("dataset_test.csv")

# 2. SELECCIÓN DE FEATURES Y TARGET
# Eliminamos 'target' y 'pdb_id' de las X para que el modelo no haga "trampas"
X = df.drop(columns=['target', 'pdb_id'])
#la variable Y es lo que intentamos predecir
y = df['target']

# 3. SPLIT DE DATOS (80% entrenamiento, 20% test)
# random_state asegura que si repites el código, los resultados sean iguales
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  #random_state es el set.seed de Python

#Separar por proteinas no por puntos.

groups = df["pdb_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print("Proteínas en train:", len(set(df.iloc[train_idx]["pdb_id"])))
print("Proteínas en test:", len(set(df.iloc[test_idx]["pdb_id"])))

# comprobar que no hay overlap
train_prots = set(df.iloc[train_idx]["pdb_id"])
test_prots = set(df.iloc[test_idx]["pdb_id"])

print("Overlap:", len(train_prots & test_prots))

# FASE 2: Entrenamiento final con parámetros ya conocidos
mejor_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=2
)

print("Entrenando modelo final con X_train completo...")
mejor_rf.fit(X_train, y_train)
print("Entrenamiento completado.")

# EVALUACIÓN FINAL
y_proba = mejor_rf.predict_proba(X_test)[:, 1]
threshold = 0.5 #con 0.7 se reduce el número de falsos positivos pero también los verdaderos positivos (muchos falsos negativos)
y_pred = (y_proba >= threshold).astype(int)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
#print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")
#print(f"MCC:     {matthews_corrcoef(y_test, y_pred):.4f}")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

joblib.dump(mejor_rf, 'modelo_rf_predictor.pkl')
print("Modelo guardado.")

# MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Bolsillo (0)', 'Bolsillo (1)'],
            yticklabels=['No Bolsillo (0)', 'Bolsillo (1)'])
plt.title('Matriz de Confusión — Modelo Final')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("Prob media:", np.mean(y_proba))
print("Max prob:", np.max(y_proba))


# IMPORTANCIA DE FEATURES
importancias = pd.Series(
    mejor_rf.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n--- Importancia de las Características ---")
print(importancias.head(20))

