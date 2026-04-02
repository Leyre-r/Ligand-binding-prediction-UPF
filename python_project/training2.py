#Data procesing
import pandas as pd

#Data Modelling
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score #otras posibles metricas: precision_score, recall_score, ConfusionMatrixDisplay
import joblib

# 1. CARGAR DATOS
df = pd.read_csv("dataset_training_completo.csv")

# 2. SELECCIÓN DE FEATURES Y TARGET
# Eliminamos 'target' y 'pdb_id' de las X para que el modelo no haga "trampas"
X = df.drop(columns=['target', 'pdb_id'])
#la variable Y es lo que intentamos predecir
y = df['target']

# 3. SPLIT DE DATOS (80% entrenamiento, 20% test)
# random_state asegura que si repites el código, los resultados sean iguales
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  #random_state es el set.seed de Python

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
y_pred  = mejor_rf.predict(X_test)
y_proba = mejor_rf.predict_proba(X_test)[:, 1]

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
#print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")
#print(f"MCC:     {matthews_corrcoef(y_test, y_pred):.4f}")

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



# IMPORTANCIA DE FEATURES
importancias = pd.Series(
    mejor_rf.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n--- Importancia de las Características ---")
print(importancias.head(20))

