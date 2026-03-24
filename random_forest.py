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

# 4. CONFIGURACIÓN DEL MODELO
# Usamos class_weight='balanced' porque hay muchísimos más 0s que 1s
rf = RandomForestClassifier(
    class_weight='balanced', # ¡CRÍTICO para bioinformática!
    n_jobs=2,              # NO Usa todos los núcleos deL procesador
    random_state=42
)

# Definimos el espacio de búsqueda
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Añadimos el Cross-Validation=5
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=1,      #Mientras hago las primeras pruebas lo he bajado porque no arrancaba
    scoring='roc_auc',
    n_jobs=-1,
    verbose=3
)

print("Iniciando GridSearchCV y Cross-Validation...")
grid_search.fit(X_train, y_train)

# EXTRAEMOS EL MEJOR MODELO (Este ya está entrenado con los mejores parámetros)
mejor_rf = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")

# 5. EVALUACIÓN FINAL (Usando el set de Test que el modelo nunca vio)
y_pred = mejor_rf.predict(X_test)
y_proba = mejor_rf.predict_proba(X_test)[:, 1]

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score Final: {roc_auc_score(y_test, y_proba):.2f}")

# 6. MATRIZ DE CONFUSIÓN 
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Bolsillo (0)', 'Bolsillo (1)'], 
            yticklabels=['No Bolsillo (0)', 'Bolsillo (1)'])
plt.title('Matriz de Confusión: Mejor Modelo Optimizado')
plt.show()

# 7. IMPORTANCIA DE LAS FEATURES (Del mejor modelo)
importancias = pd.Series(mejor_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Importancia de las Características ---")
print(importancias)

# 8. GUARDAR EL MEJOR MODELO
joblib.dump(mejor_rf, 'modelo_rf_predictor.pkl')

# --- VALIDACIÓN INDEPENDIENTE ---
# Supongamos que has procesado una cohorte independiente y la tienes en 'independent_test.csv'
#df_independent = pd.read_csv("independent_test.csv")
#X_indep = df_independent.drop(columns=['target', 'pdb_id'], errors='ignore')
#y_indep = df_independent['target']

# Evaluar el modelo que entrenaste con PDBbind en esta nueva cohorte
#y_pred_indep = mejor_rf.predict(X_indep)

#print("\n--- RENDIMIENTO EN COHORTE INDEPENDIENTE ---")
#print(classification_report(y_indep, y_pred_indep))
#print(f"ROC AUC Independiente: {roc_auc_score(y_indep, mejor_rf.predict_proba(X_indep)[:, 1]):.2f}")


