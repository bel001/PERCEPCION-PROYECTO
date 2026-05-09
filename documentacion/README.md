# Sistema de Vision Artificial para Deteccion de Equipo de Protección Personal con YOLO

## 1. Problematica

La supervision manual del Equipo de Protección Personal, como cascos,
chalecos, guantes, mascarillas, gafas o zapatos de seguridad, es lenta y puede
fallar por cansancio, puntos ciegos o descuido humano. En zonas industriales y
de construccion, esas fallas elevan el riesgo de accidentes laborales porque
una infraccion puede pasar desapercibida justo antes de una actividad peligrosa.

## 2. Solucion Propuesta

El proyecto implementa un sistema de vision artificial con YOLO para detectar
trabajadores y verificar si usan el Equipo de Protección Personal obligatorio. El sistema puede trabajar
con imagenes, videos o webcam. Cuando detecta una persona sin el equipo requerido,
dibuja una alerta visual sobre el fotograma.

El Equipo de Protección Personal obligatorio no esta fijo en el codigo. Se configura en:

```text
epp-yolo/configs/reglas_epp.yaml
```

Por defecto se exige:

```yaml
epp_obligatorio:
  - helmet
  - vest
```

## 3. Estructura del Proyecto

```text
PERCEPCION/
├── documentacion/
│   └── README.md
└── epp-yolo/
    ├── configs/
    │   └── reglas_epp.yaml
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── outputs/
    ├── src/
    │   ├── configuracion.py
    │   ├── preprocesar.py
    │   ├── entrenar.py
    │   ├── validar.py
    │   └── detectar.py
    ├── main.py
    ├── requirements.txt
    └── README.md
```

## 4. Fuentes de Datos

Los datasets se descargan manualmente porque algunos requieren cuenta, token,
aceptacion de licencia o exportacion desde la plataforma. Coloca los datos
descargados dentro de:

```text
epp-yolo/data/raw/
```

Estado actual del proyecto: el dataset no esta incluido en la carpeta. La
carpeta `epp-yolo/data/raw/` esta vacia y sirve como destino para pegar ahi los
datasets descargados. La carpeta `epp-yolo/data/processed/` se genera despues de
ejecutar la limpieza.

Fuentes recomendadas:

- Hugging Face PPE Detection: https://huggingface.co/datasets/51ddhesh/PPE_Detection
- Mendeley PPE 5-Class: https://data.mendeley.com/datasets/8vf7z6v5sb/1
- Roboflow Hard Hat Universe: https://universe.roboflow.com/ppe-demo/hard-hat-universe-0dy7t-enpal
- Mendeley Dataset of PPE: https://data.mendeley.com/datasets/zkzghjvpn2
- CHV / Real-time PPE dataset: https://github.com/ZijianWang-ZW/PPE_detection
- Ultralytics YOLO: https://docs.ultralytics.com/

Peso aproximado de los datasets:

| Fuente | Peso aproximado | Nota |
| --- | ---: | --- |
| Hugging Face PPE Detection | 668 MB | Archivo principal `PPE.zip`. |
| Mendeley PPE 5-Class | 124 MB | Total aproximado sumando archivos publicos de la version 1. |
| Roboflow Hard Hat Universe | No publicado | La pagina indica 7,036 imagenes; el peso depende del formato de exportacion. |
| Mendeley Dataset of PPE | 236 MB | Archivo `20250731-PPE2286y.zip`. |
| CHV / Real-time PPE dataset | 440 MB | Archivo `CHV_dataset.zip` desde Google Drive. |

En total, sin contar Roboflow, se debe reservar aproximadamente 1.47 GB solo
para archivos comprimidos. Descomprimidos y preprocesados pueden ocupar mas
espacio.

Recomendacion practica: combinar fuentes ayuda a mejorar la efectividad, pero
solo despues de normalizar nombres de clases. Por ejemplo, distintos datasets
pueden usar `hardhat`, `safety_helmet`, `helmet` o `head with helmet` para la
misma idea. El archivo `reglas_epp.yaml` contiene alias para unificar esas clases.

## 5. Instalacion

Desde la carpeta del proyecto:

```bash
cd epp-yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El archivo `requirements.txt` instala PyTorch para CPU por defecto para evitar
descargas CUDA innecesarias en equipos sin GPU NVIDIA. Si tienes GPU NVIDIA
configurada, instala PyTorch con CUDA desde la guia oficial de PyTorch y luego
instala el resto de dependencias.

## 6. Comandos para Ejecutar el Proyecto

Todos los comandos se ejecutan desde la carpeta `epp-yolo`:

```bash
cd /home/rodrigo/Escritorio/Codigos/PERCEPCION/epp-yolo
source .venv/bin/activate
```

No ejecutes `main.py` sin indicar accion. Este comando esta incompleto:

```bash
python main.py
```

Debe incluir un subcomando:

```bash
python main.py camara
python main.py validar
python main.py entrenar
python main.py detectar --fuente ruta/imagen.jpg
```

Tampoco uses `/bin/python3.13` directamente, porque ese Python no tiene las
dependencias del proyecto. Usa el Python del entorno virtual activando `.venv`:

```bash
cd /home/rodrigo/Escritorio/Codigos/PERCEPCION/epp-yolo
source .venv/bin/activate
python main.py camara
```

Si el entorno no existe:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Descomprimir el dataset descargado:

```bash
unzip data/raw/huggingface_ppe/PPE.zip -d data/raw/huggingface_ppe_extraido
```

Preprocesar y limpiar imagenes:

```bash
python main.py preprocesar --entrada data/raw/huggingface_ppe_extraido --salida data/processed --limpiar-ruido --sobrescribir
```

Entrenar rapido para probar el flujo:

```bash
python main.py entrenar --epocas 3 --tamano 416 --lote 4
```

Entrenar con mejor calidad, pero mas lento:

```bash
python main.py entrenar --epocas 50 --tamano 640 --lote 8
```

Validar el modelo entrenado:

```bash
python main.py validar
```

Abrir la camara con la vista de deteccion:

```bash
python main.py camara
```

Detectar una imagen:

```bash
python main.py detectar --fuente ruta/imagen.jpg
```

Detectar un video y guardar el resultado:

```bash
python main.py detectar --fuente ruta/video.mp4 --guardar
```

Si se desea usar directamente el modelo entrenado actual, su ruta es:

```text
runs/detect/runs/detect/epp_train-2/weights/best.pt
```

Comando directo equivalente para abrir camara con ese modelo:

```bash
python -m src.detectar --modelo runs/detect/runs/detect/epp_train-2/weights/best.pt --fuente 0 --dispositivo cpu
```

## 7. Capas del Pipeline

El proyecto esta organizado por capas para separar cada responsabilidad del
sistema:

```text
data/raw/
  ↓
Capa 1: preprocesamiento y limpieza de imagenes
  src/preprocesar.py
  ↓
data/processed/epp.yaml
  ↓
Capa 2: entrenamiento del modelo YOLO
  src/entrenar.py
  ↓
runs/detect/epp_train/weights/best.pt
  ↓
Capa 3: validacion de metricas
  src/validar.py
  ↓
Capa 4: deteccion e inferencia en imagen, video o webcam
  src/detectar.py
  ↓
outputs/
```

Resumen de cada capa:

- Main: `main.py` centraliza los comandos principales del proyecto.
- Preprocesamiento: limpia ruido, mejora iluminacion, valida imagenes y normaliza etiquetas.
- Entrenamiento: usa `data/processed/epp.yaml` para entrenar YOLO.
- Validacion: calcula metricas como `mAP50`, `mAP75` y `mAP50-95`.
- Deteccion: usa el modelo entrenado para detectar Equipo de Protección Personal y generar alertas.

Comandos simplificados con `main.py`:

```bash
python main.py preprocesar --entrada data/raw/huggingface_ppe_extraido --salida data/processed --limpiar-ruido --sobrescribir
python main.py entrenar --epocas 3 --tamano 416 --lote 4
python main.py validar
python main.py camara
```

## 8. Limpieza de Ruido y Preparacion

El script `src/preprocesar.py` prepara los datasets antes del entrenamiento.
Realiza estas tareas:

- valida que las imagenes se puedan leer;
- descarta imagenes corruptas o sin etiqueta YOLO asociada;
- reduce ruido con filtro mediano o denoising avanzado;
- mejora contraste e iluminacion con CLAHE;
- redimensiona imagenes manteniendo proporcion;
- copia y normaliza etiquetas YOLO;
- genera `data/processed/epp.yaml` para entrenar con Ultralytics.

Comando base:

```bash
python -m src.preprocesar --entrada data/raw --salida data/processed --sobrescribir
```

Con reduccion de ruido mas fuerte:

```bash
python -m src.preprocesar --entrada data/raw --salida data/processed --limpiar-ruido --sobrescribir
```

Formato esperado por dataset:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 9. Entrenamiento

Despues de procesar datos:

```bash
python -m src.entrenar --datos data/processed/epp.yaml --modelo yolo11n.pt
```

Opciones utiles:

```bash
python -m src.entrenar \
  --datos data/processed/epp.yaml \
  --modelo yolo11n.pt \
  --epocas 80 \
  --tamano 640 \
  --lote 8 \
  --dispositivo cpu
```

El modelo entrenado quedara normalmente en:

```text
runs/detect/epp_train/weights/best.pt
```

## 10. Validacion

Para medir el rendimiento:

```bash
python -m src.validar \
  --modelo runs/detect/epp_train/weights/best.pt \
  --datos data/processed/epp.yaml
```

Metricas principales:

- `mAP50-95`: precision media en varios umbrales IoU;
- `mAP50`: precision media con IoU 0.50;
- `mAP75`: precision media con IoU 0.75.

## 11. Deteccion y Alertas

La vista de deteccion muestra:

- la imagen de la camara, video o archivo;
- cajas sobre cada objeto detectado;
- etiquetas en espanol como `trabajador`, `casco`, `chaleco`, `guantes`;
- un panel superior con el resumen: trabajadores detectados, objetos observados y estado de alerta.

Webcam:

```bash
python -m src.detectar --modelo runs/detect/epp_train/weights/best.pt --fuente 0
```

Imagen:

```bash
python -m src.detectar --modelo runs/detect/epp_train/weights/best.pt --fuente ruta/imagen.jpg
```

Video:

```bash
python -m src.detectar \
  --modelo runs/detect/epp_train/weights/best.pt \
  --fuente ruta/video.mp4 \
  --guardar
```

Carpeta de imagenes:

```bash
python -m src.detectar --modelo runs/detect/epp_train/weights/best.pt --fuente ruta/imagenes
```

Las salidas anotadas se guardan en:

```text
epp-yolo/outputs/
```

## 12. Como Funcionan las Reglas

El detector obtiene cajas de YOLO y separa:

- `person`: trabajador detectado;
- clases de Equipo de Protección Personal: `helmet`, `vest`, `gloves`, etc.;
- clases negativas: `no_helmet`, `no_vest`, etc.

Luego asocia cada elemento de Equipo de Protección Personal con la persona mas cercana por superposicion o centro de
caja. Si falta algun elemento listado en `epp_obligatorio`, se marca infraccion.

Ejemplo para exigir casco, chaleco y guantes:

```yaml
epp_obligatorio:
  - helmet
  - vest
  - gloves
```

## 13. Pruebas Recomendadas

1. Probar limpieza con un dataset pequeno:

   ```bash
   python -m src.preprocesar --entrada data/raw --salida data/processed --sobrescribir
   ```

2. Confirmar que existen imagenes y etiquetas:

   ```bash
   find data/processed -type f | head
   ```

3. Entrenar pocas epocas para validar el pipeline:

   ```bash
   python -m src.entrenar --datos data/processed/epp.yaml --modelo yolo11n.pt --epocas 1
   ```

4. Validar metricas:

   ```bash
   python -m src.validar --modelo runs/detect/epp_train/weights/best.pt --datos data/processed/epp.yaml
   ```

5. Probar deteccion con imagen, video y webcam.

6. Cambiar `epp_obligatorio` en `reglas_epp.yaml` y confirmar que la alerta cambia
   sin modificar codigo Python.

## 14. Limitaciones

- Si el dataset no incluye la clase `person`, el sistema puede detectar Equipo de Protección Personal pero
  no podra asignar correctamente infracciones a trabajadores.
- La deteccion en tiempo real depende de CPU/GPU, resolucion, modelo usado y FPS
  de la camara.
- Mezclar datasets mejora cobertura, pero puede introducir ruido si las clases
  no se normalizan bien.
- Las alertas visuales son apoyo preventivo; no reemplazan protocolos formales
  de seguridad ocupacional.

## 15. Mejoras Futuras

- Agregar seguimiento de personas entre frames para reducir falsos positivos.
- Exportar a ONNX o TensorRT para despliegue en edge devices.
- Registrar eventos de infraccion en CSV o base de datos.
- Crear una interfaz web para cargar videos e inspeccionar resultados.
- Ajustar reglas por zona de trabajo, por ejemplo casco obligatorio en una zona
  y guantes obligatorios en otra.
