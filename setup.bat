@echo off
echo ================================================
echo  CONFIGURACION APLICACION WEB - REGRESION LOGISTICA
echo ================================================
echo.

echo [1/4] Instalando dependencias...
pip install -r requirements.txt

echo.
echo [2/4] Verificando dataset...
if exist "UserData.csv" (
    echo ✓ Dataset UserData.csv encontrado
) else (
    echo ✗ Error: UserData.csv no encontrado
    echo   Asegurate de que el archivo este en esta carpeta
    pause
    exit /b 1
)

echo.
echo [3/4] Entrenando modelo de machine learning...
python train_model.py

echo.
echo [4/4] Iniciando aplicacion web...
echo.
echo ================================================
echo  APLICACION LISTA!
echo ================================================
echo  Abre tu navegador y ve a: http://localhost:5000
echo ================================================
echo.

python app.py