#  Predictor de Precios de Viviendas - Melbourne

Proyecto de Machine Learning que predice precios de propiedades en Melbourne, Australia, utilizando un modelo de Árboles de Decisión.

##  Características Principales
- Modelo entrenado con 1,300+ registros reales
- API REST con Flask para predicciones en tiempo real
- Interfaz web intuitiva
- Métricas de rendimiento del modelo (MAE, R²)

##  Instalación y Ejecución

### Requisitos Previos
- Python 3.1+

### Pasos para Configuración

1. **Clonar repositorio**:
```bash
git clone https://github.com/Luxycotin/TLP3_Evaluativo
```

2. **Configurar entorno virtual:**
```bash
python -m venv venv
---------------------
venv\Scripts\activate

```

3. **Instalar dependencias:**
```bash
pip install 
``` 

###  Entrenar el Modelo

```bash
Ejecutar models/train_model.py
```

### Iniciar la API

```bash
python app.py
```

###  Ejecutar Cliente Web

1. Abrir **client/index.html** en tu navegador
2. Usar **Live Server** (extensión de VS Code) para mejor experiencia