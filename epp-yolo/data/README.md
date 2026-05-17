# Carpeta de Datos

Esta carpeta guarda datasets locales, pero GitHub no los versiona porque son
archivos pesados. Solo se sube este README y los `.gitkeep` necesarios para
mantener la estructura.

## Dataset principal: SH17

Fuente:

```text
https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
```

En esta maquina SH17 esta organizado asi:

```text
data/raw/sh17/sh17_kaggle.zip              # descarga original de Kaggle
data/raw/sh17/original/images/             # imagenes extraidas
data/raw/sh17/original/labels/             # etiquetas YOLO originales
data/raw/sh17/original/train_files.txt     # lista de entrenamiento
data/raw/sh17/original/val_files.txt       # lista de validacion
data/raw/sh17/yolo/                        # estructura YOLO train/valid
data/processed/sh17/epp.yaml               # YAML recomendado para entrenar
```

El archivo que debe usarse en entrenamiento es:

```text
data/processed/sh17/epp.yaml
```

## Si se descarga otra vez

Descarga SH17 desde Kaggle y deja el contenido extraido con esta forma:

```text
data/raw/sh17/original/
├── images/
├── labels/
├── train_files.txt
└── val_files.txt
```

Luego reconstruye la estructura YOLO y el dataset normalizado:

```bash
python main.py preparar-sh17 --sobrescribir
```

Verifica el resultado:

```bash
python main.py revisar-dataset --ruta data/processed/sh17
```

## Estructura YOLO esperada

Para cualquier otro dataset YOLO, la estructura minima es:

```text
data/raw/nombre_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

Si un dataset trae `train/images`, `train/labels`, `valid/images` y
`valid/labels`, pero no trae YAML, puedes generar uno base:

```bash
python main.py revisar-dataset --ruta data/raw/nombre_dataset --crear-yaml
```

## Entrenamiento con SH17

Prueba rapida:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 3 --tamano 416 --lote 4 --dispositivo cpu --nombre epp_sh17_prueba
```

Entrenamiento principal:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 50 --tamano 640 --lote 8 --dispositivo cpu --nombre epp_sh17
```
