# Deteccion de Equipo de Protección Personal con YOLO

Proyecto Python para detectar trabajadores y Equipo de Protección Personal en
imagenes, videos y webcam usando YOLO.

La documentacion completa esta en:

```text
../documentacion/README.md
```

## Dataset SH17

Dataset recomendado:

```text
https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
```

El dataset no se sube a GitHub por peso. En esta maquina quedo localmente asi:

```text
data/raw/sh17/sh17_kaggle.zip
data/raw/sh17/original/images/
data/raw/sh17/original/labels/
data/raw/sh17/original/train_files.txt
data/raw/sh17/original/val_files.txt
data/raw/sh17/yolo/
data/processed/sh17/epp.yaml
```

`data/processed/sh17/epp.yaml` es el YAML recomendado para entrenar porque
normaliza las clases de SH17 al formato del proyecto:

```text
person, helmet, vest, gloves, goggles, mask, safety_shoes, no_helmet, no_vest
```

Para reconstruir `data/raw/sh17/yolo` y `data/processed/sh17` desde la carpeta
extraida:

```bash
python main.py preparar-sh17 --sobrescribir
```

Para comprobar que el dataset esta listo:

```bash
python main.py revisar-dataset --ruta data/processed/sh17
```

## Instalacion

```bash
cd epp-yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si ya existe `.venv`, `main.py` intenta relanzarse automaticamente con el Python
del entorno virtual, pero lo recomendado es activarlo manualmente.

## Comandos principales

`main.py` siempre necesita un subcomando. Este comando esta incompleto:

```bash
python main.py
```

Usa uno de estos:

```bash
python main.py preparar-sh17
python main.py revisar-dataset --ruta data/processed/sh17
python main.py entrenar
python main.py validar
python main.py evaluar
python main.py detectar --fuente ruta/imagen.jpg
python main.py camara
```

Entrenamiento rapido para probar el flujo:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 3 --tamano 416 --lote 4 --dispositivo cpu --nombre epp_sh17_prueba
```

Entrenamiento principal:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 50 --tamano 640 --lote 8 --dispositivo cpu --nombre epp_sh17
```

Validar metricas:

```bash
python main.py validar --modelo runs/detect/epp_sh17/weights/best.pt --datos data/processed/sh17/epp.yaml --dispositivo cpu
```

Evaluar y guardar reporte de error:

```bash
python main.py evaluar --modelo runs/detect/epp_sh17/weights/best.pt --datos data/processed/sh17/epp.yaml --dispositivo cpu --salida outputs/evaluacion_modelo.json
```

Probar con camara:

```bash
python main.py camara --modelo runs/detect/epp_sh17/weights/best.pt
```

Si la camara va lenta en CPU:

```bash
python main.py camara --modelo runs/detect/epp_sh17/weights/best.pt --tamano 320 --saltar-frames 3 --ancho-camara 640 --alto-camara 480 --ancho-ventana 800 --alto-ventana 600
```

Si la camara sale de lado:

```bash
python main.py camara --modelo runs/detect/epp_sh17/weights/best.pt --rotar 90
```

## Capas principales

- `main.py`: punto de entrada para ejecutar todo con comandos simples.
- `src/preparar_sh17.py`: prepara SH17 en estructura YOLO y normaliza etiquetas.
- `src/revisar_dataset.py`: diagnostica si un dataset esta listo para YOLO.
- `src/preprocesar.py`: limpia ruido, mejora contraste y normaliza etiquetas.
- `src/entrenar.py`: entrena el modelo YOLO.
- `src/validar.py`: valida metricas y genera reporte de error aproximado.
- `src/detectar.py`: detecta en imagen, video o webcam con cajas y alertas.

## Reglas de seguridad

El Equipo de Protección Personal obligatorio se configura en:

```text
configs/reglas_epp.yaml
```

Por defecto se exige casco y chaleco:

```yaml
epp_obligatorio:
  - helmet
  - vest
```

Los datasets, resultados, pesos `.pt`, `outputs/`, `runs/` y `.venv/` quedan
fuera de Git para mantener el repositorio liviano.
