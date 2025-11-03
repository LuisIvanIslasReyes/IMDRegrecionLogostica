"""
Script para entrenar el modelo de regresión logística basado en UserData.csv
Basado en el notebook Lab04_9B_LogisticRegressionErnesto.ipynb
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                           recall_score, f1_score, classification_report, 
                           roc_curve, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

class ModeloRegresionLogistica:
    def __init__(self, dataset_path='UserData.csv'):
        self.dataset_path = dataset_path
        self.modelo = None
        self.scaler = None
        self.metricas = {}
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
    def cargar_datos(self):
        """Cargar y preparar los datos"""
        print("Cargando dataset...")
        self.dataUser = pd.read_csv(self.dataset_path)
        
        # Extraer características (Edad y Salario)
        self.x = self.dataUser.iloc[:, [2, 3]].values
        # Extraer variable objetivo (Purchased)
        self.y = self.dataUser.iloc[:, 4].values
        
        print(f"Datos cargados: {self.dataUser.shape[0]} registros")
        print(f"Características: {self.dataUser.columns[2:4].tolist()}")
        
    def dividir_datos(self, test_size=0.2, random_state=42):
        """Dividir datos en entrenamiento y prueba"""
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Datos divididos - Entrenamiento: {len(self.x_train)}, Prueba: {len(self.x_test)}")
        
    def escalar_datos(self):
        """Estandarizar las características"""
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        print("Datos escalados correctamente")
        
    def entrenar_modelo(self):
        """Entrenar el modelo de regresión logística"""
        self.modelo = LogisticRegression(random_state=42)
        self.modelo.fit(self.x_train, self.y_train)
        print("Modelo entrenado correctamente")
        
    def hacer_predicciones(self):
        """Realizar predicciones y calcular métricas"""
        self.y_pred = self.modelo.predict(self.x_test)
        self.y_pred_proba = self.modelo.predict_proba(self.x_test)[:, 1]
        
        # Calcular métricas
        self.metricas = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1_score': f1_score(self.y_test, self.y_pred),
            'auc_score': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        # Matriz de confusión
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        
        print("Predicciones realizadas y métricas calculadas")
        
    def mostrar_metricas(self):
        """Mostrar las métricas del modelo"""
        print("\n=== MÉTRICAS DEL MODELO ===")
        print(f"Accuracy: {self.metricas['accuracy']:.4f} ({self.metricas['accuracy']*100:.2f}%)")
        print(f"Precision: {self.metricas['precision']:.4f}")
        print(f"Recall: {self.metricas['recall']:.4f}")
        print(f"F1-Score: {self.metricas['f1_score']:.4f}")
        print(f"AUC Score: {self.metricas['auc_score']:.4f}")
        
        print("\n=== MATRIZ DE CONFUSIÓN ===")
        tn, fp, fn, tp = self.cm.ravel()
        print(f"Verdaderos Negativos (TN): {tn}")
        print(f"Falsos Positivos (FP): {fp}")
        print(f"Falsos Negativos (FN): {fn}")
        print(f"Verdaderos Positivos (TP): {tp}")
        
    def guardar_modelo(self, modelo_path='modelo_regresion_logistica.pkl', 
                      scaler_path='scaler.pkl', metricas_path='metricas.pkl'):
        """Guardar modelo, scaler y métricas"""
        joblib.dump(self.modelo, modelo_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Guardar métricas y datos adicionales
        datos_modelo = {
            'metricas': self.metricas,
            'matriz_confusion': self.cm.tolist(),
            'coeficientes': self.modelo.coef_[0].tolist(),
            'intercepto': self.modelo.intercept_[0],
            'caracteristicas': ['Edad', 'Salario Estimado']
        }
        joblib.dump(datos_modelo, metricas_path)
        
        print(f"Modelo guardado como: {modelo_path}")
        print(f"Scaler guardado como: {scaler_path}")
        print(f"Métricas guardadas como: {metricas_path}")
        
    def entrenar_modelo_completo(self):
        """Ejecutar todo el proceso de entrenamiento"""
        self.cargar_datos()
        self.dividir_datos()
        self.escalar_datos()
        self.entrenar_modelo()
        self.hacer_predicciones()
        self.mostrar_metricas()
        self.guardar_modelo()
        
        return self.metricas

# Función para predecir con nuevos datos
def predecir_compra(edad, salario, modelo_path='modelo_regresion_logistica.pkl', 
                   scaler_path='scaler.pkl'):
    """
    Predice si un usuario comprará basado en edad y salario
    """
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    
    # Escalar los nuevos datos
    nuevo_dato = np.array([[edad, salario]])
    nuevo_dato_escalado = scaler.transform(nuevo_dato)
    
    # Predecir clase y probabilidad
    clase_predicha = modelo.predict(nuevo_dato_escalado)[0]
    probabilidad = modelo.predict_proba(nuevo_dato_escalado)[0][1]
    
    return clase_predicha, probabilidad

if __name__ == "__main__":
    # Crear y entrenar el modelo
    modelo_lr = ModeloRegresionLogistica()
    metricas = modelo_lr.entrenar_modelo_completo()
    
    # Probar predicciones
    print("\n=== PRUEBAS DE PREDICCIÓN ===")
    ejemplos = [(25, 40000), (35, 80000), (45, 120000), (30, 30000)]
    
    for edad, salario in ejemplos:
        clase, prob = predecir_compra(edad, salario)
        decision = "COMPRARÁ" if clase == 1 else "NO COMPRARÁ"
        print(f"Edad: {edad}, Salario: ${salario:,} → {decision} (prob: {prob:.2%})")