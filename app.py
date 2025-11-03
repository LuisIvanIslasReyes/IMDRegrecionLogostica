"""
Aplicación web Flask para mostrar información del modelo de regresión logística
"""

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import roc_curve
import json
import os

app = Flask(__name__)

class ModeloWebApp:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.datos_modelo = None
        self.dataset = None
        self.cargar_modelo()
        self.cargar_dataset()
        
    def cargar_modelo(self):
        """Cargar modelo entrenado y datos"""
        try:
            self.modelo = joblib.load('modelo_regresion_logistica.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.datos_modelo = joblib.load('metricas.pkl')
            print("Modelo cargado correctamente")
        except FileNotFoundError:
            print("Archivos del modelo no encontrados. Ejecuta train_model.py primero.")
            
    def cargar_dataset(self):
        """Cargar dataset original"""
        try:
            self.dataset = pd.read_csv('UserData.csv')
        except FileNotFoundError:
            print("Dataset UserData.csv no encontrado")
            
    def predecir(self, edad, salario):
        """Hacer predicción para nuevos datos"""
        if self.modelo is None or self.scaler is None:
            return None, None
            
        nuevo_dato = np.array([[edad, salario]])
        nuevo_dato_escalado = self.scaler.transform(nuevo_dato)
        
        clase_predicha = self.modelo.predict(nuevo_dato_escalado)[0]
        probabilidad = self.modelo.predict_proba(nuevo_dato_escalado)[0][1]
        
        return clase_predicha, probabilidad
        
    def generar_grafico_confusion_matrix(self):
        """Generar gráfico de matriz de confusión"""
        if self.datos_modelo is None:
            return None
            
        plt.figure(figsize=(8, 6))
        cm = np.array(self.datos_modelo['matriz_confusion'])
        
        # Crear colormap personalizado azul grisáceo
        colors = ['#f8f9fa', '#546e7a']
        n_bins = 100
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('blue_grey', colors, N=n_bins)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                   xticklabels=['No Compra', 'Compra'],
                   yticklabels=['No Compra', 'Compra'])
        plt.title('Matriz de Confusión', fontsize=14, color='#37474f')
        plt.ylabel('Predicción', color='#546e7a')
        plt.xlabel('Real', color='#546e7a')
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    def generar_grafico_distribucion(self):
        """Generar gráfico de distribución de datos"""
        if self.dataset is None:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Colores azul grisáceo
        color_no_compra = '#90a4ae'
        color_compra = '#546e7a'
        
        # Distribución por edad
        self.dataset[self.dataset['Purchased'] == 0]['Age'].hist(
            alpha=0.7, label='No Compra', ax=ax1, bins=20, color=color_no_compra)
        self.dataset[self.dataset['Purchased'] == 1]['Age'].hist(
            alpha=0.8, label='Compra', ax=ax1, bins=20, color=color_compra)
        ax1.set_title('Distribución por Edad', fontsize=12, color='#37474f')
        ax1.set_xlabel('Edad', color='#546e7a')
        ax1.set_ylabel('Frecuencia', color='#546e7a')
        ax1.legend()
        
        # Distribución por salario
        self.dataset[self.dataset['Purchased'] == 0]['EstimatedSalary'].hist(
            alpha=0.7, label='No Compra', ax=ax2, bins=20, color=color_no_compra)
        self.dataset[self.dataset['Purchased'] == 1]['EstimatedSalary'].hist(
            alpha=0.8, label='Compra', ax=ax2, bins=20, color=color_compra)
        ax2.set_title('Distribución por Salario', fontsize=12, color='#37474f')
        ax2.set_xlabel('Salario Estimado', color='#546e7a')
        ax2.set_ylabel('Frecuencia', color='#546e7a')
        ax2.legend()
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    def obtener_estadisticas_dataset(self):
        """Obtener estadísticas del dataset"""
        if self.dataset is None:
            return {}
            
        stats = {
            'total_registros': len(self.dataset),
            'compradores': len(self.dataset[self.dataset['Purchased'] == 1]),
            'no_compradores': len(self.dataset[self.dataset['Purchased'] == 0]),
            'edad_promedio': self.dataset['Age'].mean(),
            'salario_promedio': self.dataset['EstimatedSalary'].mean(),
            'porcentaje_compra': (len(self.dataset[self.dataset['Purchased'] == 1]) / len(self.dataset)) * 100
        }
        
        return stats

# Instancia global de la aplicación del modelo
modelo_app = ModeloWebApp()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/modelo')
def modelo():
    """Página de información del modelo"""
    if modelo_app.datos_modelo is None:
        return render_template('error.html', 
                             mensaje="Modelo no encontrado. Ejecuta train_model.py primero.")
    
    metricas = modelo_app.datos_modelo['metricas']
    cm_plot = modelo_app.generar_grafico_confusion_matrix()
    dist_plot = modelo_app.generar_grafico_distribucion()
    stats = modelo_app.obtener_estadisticas_dataset()
    
    return render_template('modelo.html', 
                         metricas=metricas,
                         cm_plot=cm_plot,
                         dist_plot=dist_plot,
                         stats=stats)

@app.route('/prediccion')
def prediccion():
    """Página de predicción"""
    return render_template('prediccion.html')

@app.route('/api/predecir', methods=['POST'])
def api_predecir():
    """API para hacer predicciones"""
    try:
        data = request.get_json()
        edad = float(data['edad'])
        salario = float(data['salario'])
        
        clase, probabilidad = modelo_app.predecir(edad, salario)
        
        if clase is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
            
        resultado = {
            'clase': int(clase),
            'probabilidad': float(probabilidad),
            'decision': 'COMPRARÁ' if clase == 1 else 'NO COMPRARÁ',
            'confianza': f"{probabilidad:.1%}"
        }
        
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/estadisticas')
def api_estadisticas():
    """API para obtener estadísticas"""
    stats = modelo_app.obtener_estadisticas_dataset()
    return jsonify(stats)

if __name__ == '__main__':
    # Verificar si existen los archivos del modelo
    archivos_requeridos = ['modelo_regresion_logistica.pkl', 'scaler.pkl', 'metricas.pkl']
    archivos_faltantes = [archivo for archivo in archivos_requeridos if not os.path.exists(archivo)]
    
    if archivos_faltantes:
        print("Archivos del modelo faltantes:", archivos_faltantes)
        print("Ejecuta 'python train_model.py' para entrenar el modelo primero")
    
    app.run(debug=True, host='0.0.0.0', port=5000)