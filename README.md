# Aplicación Web de Regresión Logística

Esta aplicación web muestra información detallada de un modelo de regresión logística entrenado para predecir comportamiento de compra basado en edad y salario.

## Características

- **Dashboard del Modelo**: Visualiza métricas de rendimiento, matriz de confusión y estadísticas del dataset
- **Predicciones en Tiempo Real**: Ingresa edad y salario para obtener predicciones instantáneas
- **Interfaz Atractiva**: Diseño moderno y responsivo con Bootstrap
- **Análisis Completo**: Accuracy, Precision, Recall, F1-Score y AUC

## Instalación

### 1. Instalar dependencias

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Entrenar el modelo

```bash
python train_model.py
```

### 3. Ejecutar la aplicación web

```bash
python app.py
```

### 4. Abrir en el navegador

Navega a: http://localhost:5000

## Estructura del Proyecto

```
├── UserData.csv                    # Dataset original
├── ModelodeEntrenamiento.py                  # Script para entrenar el modelo
├── app.py                         # Aplicación Flask
├── templates/                     # Plantillas HTML
│   ├── base.html                 # Plantilla base
│   ├── index.html               # Página principal
│   ├── modelo.html              # Información del modelo
│   ├── prediccion.html          # Página de predicciones
│   └── error.html               # Página de error
├── static/                       # Archivos estáticos (CSS, JS)
├── modelo_regresion_logistica.pkl # Modelo entrenado
├── scaler.pkl                    # Escalador de datos
└── metricas.pkl                  # Métricas del modelo
```

## Uso

### 1. Página Principal
- Información general sobre el modelo
- Explicación del dataset
- Navegación a otras secciones

### 2. Información del Modelo
- Métricas de rendimiento (Accuracy, Precision, Recall, AUC)
- Matriz de confusión visualizada
- Distribución de datos por edad y salario
- Estadísticas del dataset

### 3. Predicciones
- Formulario para ingresar edad y salario
- Ejemplos predefinidos para pruebas rápidas
- Historial de predicciones recientes
- Resultado con probabilidad y explicación

## API Endpoints

- `GET /` - Página principal
- `GET /modelo` - Información del modelo
- `GET /prediccion` - Página de predicciones
- `POST /api/predecir` - API para hacer predicciones
- `GET /api/estadisticas` - API para obtener estadísticas

## Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Machine Learning**: Scikit-learn
- **Visualización**: Matplotlib, Seaborn
- **Datos**: Pandas, NumPy

## Personalización

Puedes personalizar la aplicación modificando:

- **Estilos**: Edita los archivos en `static/` o las clases CSS en las plantillas
- **Colores**: Cambia los gradientes y colores en `base.html`
- **Métricas**: Agrega nuevas métricas en `train_model.py` y `app.py`
- **Gráficos**: Modifica las funciones de visualización en `app.py`

## Notas Importantes

1. **Requisitos**: Asegúrate de que el archivo `UserData.csv` esté en la carpeta raíz
2. **Entrenamiento**: Ejecuta `train_model.py` antes de usar la aplicación web
3. **Puerto**: La aplicación corre en el puerto 5000 por defecto
4. **Desarrollo**: Usa `debug=True` en `app.py` para desarrollo

## Solución de Problemas

### Error: Archivos del modelo no encontrados
- Ejecuta `python train_model.py` para entrenar el modelo

### Error: ModuleNotFoundError
- Instala las dependencias: `pip install -r requirements.txt`

### Error: Puerto en uso
- Cambia el puerto en `app.py`: `app.run(port=5001)`

## Licencia

Este proyecto es de uso educativo y puede ser modificado libremente.