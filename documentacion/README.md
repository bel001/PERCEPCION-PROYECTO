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
    │   ├── revisar_dataset.py
    │   ├── preparar_sh17.py
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

Estado actual del proyecto: el dataset no se sube a Git porque es pesado, pero
en esta maquina ya quedo descargado y preparado SH17 dentro de `epp-yolo/data/`.
Si se clona el repositorio en otra maquina, se debe descargar SH17 desde Kaggle
y dejarlo en la estructura indicada abajo.

Estructura local esperada para SH17:

```text
epp-yolo/data/raw/sh17/sh17_kaggle.zip
epp-yolo/data/raw/sh17/original/images/
epp-yolo/data/raw/sh17/original/labels/
epp-yolo/data/raw/sh17/original/train_files.txt
epp-yolo/data/raw/sh17/original/val_files.txt
epp-yolo/data/raw/sh17/yolo/
epp-yolo/data/processed/sh17/epp.yaml
```

El archivo recomendado para entrenar es:

```text
epp-yolo/data/processed/sh17/epp.yaml
```

Fuentes recomendadas:

- SH17 Dataset for PPE Detection: https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
- Hugging Face PPE Detection: https://huggingface.co/datasets/51ddhesh/PPE_Detection
- Mendeley PPE 5-Class: https://data.mendeley.com/datasets/8vf7z6v5sb/1
- Roboflow Hard Hat Universe: https://universe.roboflow.com/ppe-demo/hard-hat-universe-0dy7t-enpal
- Mendeley Dataset of PPE: https://data.mendeley.com/datasets/zkzghjvpn2
- CHV / Real-time PPE dataset: https://github.com/ZijianWang-ZW/PPE_detection
- Ultralytics YOLO: https://docs.ultralytics.com/

Peso aproximado de los datasets:

| Fuente | Peso aproximado | Nota |
| --- | ---: | --- |
| SH17 Dataset for PPE Detection | 14 GB | Dataset principal recomendado; incluye `person`, `helmet`, `safety-vest`, `gloves`, `shoes`, etc. |
| Hugging Face PPE Detection | 668 MB | Archivo principal `PPE.zip`. |
| Mendeley PPE 5-Class | 124 MB | Total aproximado sumando archivos publicos de la version 1. |
| Roboflow Hard Hat Universe | No publicado | La pagina indica 7,036 imagenes; el peso depende del formato de exportacion. |
| Mendeley Dataset of PPE | 236 MB | Archivo `20250731-PPE2286y.zip`. |
| CHV / Real-time PPE dataset | 440 MB | Archivo `CHV_dataset.zip` desde Google Drive. |

SH17 es mas pesado que las otras fuentes, pero es la mejor opcion para este
proyecto porque incluye personas/trabajadores y varios elementos de Equipo de
Protección Personal.

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
python main.py preparar-sh17
python main.py revisar-dataset --ruta data
python main.py detectar --fuente ruta/imagen.jpg
```

Si existe `.venv`, `main.py` intenta relanzarse automaticamente con el Python
del entorno virtual. Aun asi, la forma recomendada es activar `.venv`:

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

Preparar el dataset SH17 ya descargado y extraido en
`data/raw/sh17/original/`:

```bash
python main.py preparar-sh17 --sobrescribir
```

Revisar si el dataset quedo bien importado y normalizado:

```bash
python main.py revisar-dataset --ruta data/processed/sh17
```

Si el comando indica que no encontro `data.yaml/epp.yaml` ni la estructura
`train/images` + `train/labels`, la carpeta importada no esta lista para YOLO.
Debes volver a extraer o copiar la carpeta correcta del dataset.

Archivo YAML recomendado para entrenar:

```text
data/processed/sh17/epp.yaml
```

Entrenar rapido para probar el flujo:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --epocas 3 --tamano 416 --lote 4
```

Entrenar con mejor calidad, pero mas lento:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --epocas 50 --tamano 640 --lote 8
```

Validar el modelo entrenado:

```bash
python main.py validar
```

Evaluar el modelo y guardar metricas de error:

```bash
python main.py evaluar --tamano 416 --salida outputs/evaluacion_modelo.json
```

Abrir la camara con la vista de deteccion:

```bash
python main.py camara
```

Si la camara se ve lenta, usa modo rapido para CPU:

```bash
python main.py camara --tamano 320 --saltar-frames 3 --ancho-camara 640 --alto-camara 480 --ancho-ventana 800 --alto-ventana 600
```

La lentitud ocurre porque el equipo esta ejecutando YOLO en CPU. Cada frame de
la camara pasa por la red neuronal; mientras mayor sea la resolucion o el
`--tamano`, mas tarda la inferencia. `--saltar-frames 3` procesa 1 de cada 3
frames y `--tamano 320` reduce el trabajo del modelo.
El programa muestra siempre el frame actual de la camara y reutiliza las ultimas
detecciones entre frames para que la vista no se congele.

Si la imagen de la camara aparece cortada, aumenta o reduce la ventana:

```bash
python main.py camara --ancho-ventana 800 --alto-ventana 600
```

La vista mantiene la proporcion de la imagen para evitar recortes.

Si la camara aparece de lado, rota la vista:

```bash
python main.py camara --rotar 90
python main.py camara --rotar 270
```

La imagen se centra automaticamente dentro de la ventana.

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
runs/detect/epp_sh17/weights/best.pt
```

Comando directo equivalente para abrir camara con ese modelo:

```bash
python -m src.detectar --modelo runs/detect/epp_sh17/weights/best.pt --fuente 0 --dispositivo cpu
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
data/processed/sh17/epp.yaml
  ↓
Capa 2: entrenamiento del modelo YOLO
  src/entrenar.py
  ↓
runs/detect/epp_sh17/weights/best.pt
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
- Entrenamiento: usa `data/processed/sh17/epp.yaml` para entrenar YOLO.
- Validacion: calcula metricas como `mAP50`, `mAP75` y `mAP50-95`.
- Deteccion: usa el modelo entrenado para detectar Equipo de Protección Personal y generar alertas.

Comandos simplificados con `main.py`:

```bash
python main.py preparar-sh17 --sobrescribir
python main.py revisar-dataset --ruta data/processed/sh17
python main.py entrenar --datos data/processed/sh17/epp.yaml --epocas 3 --tamano 416 --lote 4
python main.py validar
python main.py evaluar
python main.py camara
python main.py camara --tamano 320 --saltar-frames 3
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
- genera un `epp.yaml` dentro de la carpeta de salida para entrenar con Ultralytics.

Comando base:

```bash
python -m src.preprocesar --entrada data/raw/sh17/yolo --salida data/processed/sh17_limpio --sobrescribir
```

Con reduccion de ruido mas fuerte:

```bash
python -m src.preprocesar --entrada data/raw/sh17/yolo --salida data/processed/sh17_limpio --limpiar-ruido --sobrescribir
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
python -m src.entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt
```

Opciones utiles:

```bash
python -m src.entrenar \
  --datos data/processed/sh17/epp.yaml \
  --modelo yolo11n.pt \
  --epocas 80 \
  --tamano 640 \
  --lote 8 \
  --dispositivo cpu
```

El modelo entrenado quedara normalmente en:

```text
runs/detect/epp_sh17/weights/best.pt
```

## 10. Validacion

Para medir el rendimiento:

```bash
python -m src.validar \
  --modelo runs/detect/epp_sh17/weights/best.pt \
  --datos data/processed/sh17/epp.yaml
```

Metricas principales:

- `mAP50-95`: precision media en varios umbrales IoU;
- `mAP50`: precision media con IoU 0.50;
- `mAP75`: precision media con IoU 0.75.

## 11. Evaluacion del Modelo y Error

Para evaluar el modelo y guardar un reporte:

```bash
python main.py evaluar --salida outputs/evaluacion_modelo.json
```

Para ver el reporte generado:

```bash
cat outputs/evaluacion_modelo.json
```

Tambien se puede ejecutar directamente:

```bash
python -m src.validar \
  --modelo runs/detect/epp_sh17/weights/best.pt \
  --datos data/processed/sh17/epp.yaml \
  --tamano 416 \
  --salida outputs/evaluacion_modelo.json
```

El reporte se guarda en:

```text
outputs/evaluacion_modelo.json
```

Metricas que muestra:

- `precision_media`: de las detecciones realizadas, cuantas fueron correctas en promedio.
- `recall_medio`: de los objetos reales, cuantos encontro el modelo en promedio.
- `mAP50`: precision media usando IoU 0.50.
- `mAP50_95`: metrica mas estricta; evalua varios umbrales IoU.
- `error_aproximado`: se calcula como `1 - metrica`.
- `metricas_por_clase`: muestra precision, recall, mAP y error para cada clase.
- `clase_con_mayor_error_aproximado`: indica la clase con peor resultado.

Ejemplo de interpretacion:

```text
mAP50 = 0.456
error aproximado = 1 - 0.456 = 0.544
```

Ese `0.544` no es un error clasico como en regresion; es una forma simple de
decir cuanto falta para llegar a 1.0 en la metrica. Mientras mas bajo sea ese
error aproximado, mejor.

Si el reporte muestra, por ejemplo, que `safety_shoes` tiene el mayor error,
significa que el modelo esta fallando mas al detectar zapatos de seguridad que
las otras clases. En ese caso conviene agregar mas imagenes de esa clase,
revisar etiquetas incorrectas o entrenar mas epocas.

## 12. Deteccion y Alertas

La vista de deteccion muestra:

- la imagen de la camara, video o archivo;
- cajas sobre cada objeto detectado;
- etiquetas en espanol como `trabajador`, `casco`, `chaleco`, `guantes`;
- un panel superior con el resumen: trabajadores detectados, objetos observados y estado de alerta.

Webcam:

```bash
python -m src.detectar --modelo runs/detect/epp_sh17/weights/best.pt --fuente 0
```

Imagen:

```bash
python -m src.detectar --modelo runs/detect/epp_sh17/weights/best.pt --fuente ruta/imagen.jpg
```

Video:

```bash
python -m src.detectar \
  --modelo runs/detect/epp_sh17/weights/best.pt \
  --fuente ruta/video.mp4 \
  --guardar
```

Carpeta de imagenes:

```bash
python -m src.detectar --modelo runs/detect/epp_sh17/weights/best.pt --fuente ruta/imagenes
```

Las salidas anotadas se guardan en:

```text
epp-yolo/outputs/
```

## 13. Como Funcionan las Reglas

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

## 14. Pruebas Recomendadas

1. Confirmar que SH17 esta listo para YOLO:

   ```bash
   python main.py revisar-dataset --ruta data/processed/sh17
   ```

2. Confirmar que existen imagenes y etiquetas. Se usa `-L` porque SH17 queda
   preparado con enlaces simbolicos para no duplicar 14 GB de imagenes:

   ```bash
   find -L data/processed/sh17 -type f | head
   ```

3. Entrenar pocas epocas para validar el pipeline:

   ```bash
   python -m src.entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 1
   ```

4. Validar metricas:

   ```bash
   python -m src.validar --modelo runs/detect/epp_sh17/weights/best.pt --datos data/processed/sh17/epp.yaml
   ```

5. Probar deteccion con imagen, video y webcam.

6. Cambiar `epp_obligatorio` en `reglas_epp.yaml` y confirmar que la alerta cambia
   sin modificar codigo Python.

## 15. Limitaciones

- Si el dataset no incluye la clase `person`, el sistema puede detectar Equipo de Protección Personal pero
  no podra asignar correctamente infracciones a trabajadores.
- La deteccion en tiempo real depende de CPU/GPU, resolucion, modelo usado y FPS
  de la camara.
- Mezclar datasets mejora cobertura, pero puede introducir ruido si las clases
  no se normalizan bien.
- Las alertas visuales son apoyo preventivo; no reemplazan protocolos formales
  de seguridad ocupacional.

## 16. Mejoras Futuras

- Agregar seguimiento de personas entre frames para reducir falsos positivos.
- Exportar a ONNX o TensorRT para despliegue en edge devices.
- Registrar eventos de infraccion en CSV o base de datos.
- Crear una interfaz web para cargar videos e inspeccionar resultados.
- Ajustar reglas por zona de trabajo, por ejemplo casco obligatorio en una zona
  y guantes obligatorios en otra.
