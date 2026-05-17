# PERCEPCION-PROYECTO

Sistema de vision artificial para detectar trabajadores y verificar el uso de
Equipo de Protección Personal con YOLO.

El proyecto incluye el flujo completo para preparar datos, entrenar, validar,
evaluar errores y probar deteccion en imagen, video o camara.

## Estructura

```text
PERCEPCION/
├── documentacion/
│   └── README.md
├── epp-yolo/
│   ├── configs/reglas_epp.yaml
│   ├── data/
│   ├── src/
│   ├── main.py
│   └── README.md
├── TECNOLOGIAS_Y_CODIGO.md
└── EXPLICACION_DETALLADA.md
```

## Dataset usado

El dataset recomendado para este proyecto es SH17:

```text
https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
```

En esta maquina ya quedo descargado y preparado localmente. No se sube a GitHub
porque pesa aproximadamente 14 GB comprimido y otros 14 GB extraido.

Rutas locales esperadas:

```text
epp-yolo/data/raw/sh17/sh17_kaggle.zip
epp-yolo/data/raw/sh17/original/images/
epp-yolo/data/raw/sh17/original/labels/
epp-yolo/data/raw/sh17/original/train_files.txt
epp-yolo/data/raw/sh17/original/val_files.txt
epp-yolo/data/processed/sh17/epp.yaml
```

El archivo que se usa para entrenar es:

```text
epp-yolo/data/processed/sh17/epp.yaml
```

## Uso rapido

```bash
cd epp-yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verificar que el dataset SH17 esta listo:

```bash
python main.py revisar-dataset --ruta data/processed/sh17
```

Entrenamiento rapido de prueba:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 3 --tamano 416 --lote 4 --dispositivo cpu --nombre epp_sh17_prueba
```

Entrenamiento principal:

```bash
python main.py entrenar --datos data/processed/sh17/epp.yaml --modelo yolo11n.pt --epocas 50 --tamano 640 --lote 8 --dispositivo cpu --nombre epp_sh17
```

Validar, evaluar y probar camara:

```bash
python main.py validar --modelo runs/detect/epp_sh17/weights/best.pt --datos data/processed/sh17/epp.yaml --dispositivo cpu
python main.py evaluar --modelo runs/detect/epp_sh17/weights/best.pt --datos data/processed/sh17/epp.yaml --dispositivo cpu
python main.py camara --modelo runs/detect/epp_sh17/weights/best.pt
```

## Documentacion

- [Manual completo](documentacion/README.md)
- [README del proyecto Python](epp-yolo/README.md)
- [Tecnologias y arquitectura](TECNOLOGIAS_Y_CODIGO.md)
- [Explicacion detallada del codigo](EXPLICACION_DETALLADA.md)

Los datasets, pesos entrenados, salidas generadas, entornos virtuales y carpetas
`runs/` no se suben al repositorio.
