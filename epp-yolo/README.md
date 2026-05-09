# Deteccion de Equipo de Protección Personal con YOLO

Proyecto Python para detectar Equipo de Protección Personal en imagenes,
videos y webcam usando YOLO.

La documentacion completa del proyecto esta en:

```text
../documentacion/README.md
```

## Uso rapido

```bash
cd epp-yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py preprocesar --entrada data/raw/huggingface_ppe_extraido --salida data/processed --limpiar-ruido --sobrescribir
python main.py entrenar --epocas 3 --tamano 416 --lote 4
python main.py validar
python main.py camara
```

`main.py` siempre necesita un subcomando. Este comando esta incompleto:

```bash
python main.py
```

Usa uno de estos:

```bash
python main.py preprocesar
python main.py entrenar
python main.py validar
python main.py detectar --fuente ruta/imagen.jpg
python main.py camara
```

Ejecuta el proyecto con el entorno `.venv` activado. No uses `/bin/python3.13`
directamente, porque ese Python no tiene las dependencias instaladas.

Tambien puedes llamar directamente los modulos, por ejemplo
`python -m src.detectar --modelo runs/detect/runs/detect/epp_train-2/weights/best.pt --fuente 0`.

## Capas principales

- `main.py`: punto de entrada principal para ejecutar todo con comandos simples.
- `src/preprocesar.py`: limpieza de ruido, mejora de contraste y normalizacion de etiquetas.
- `src/entrenar.py`: entrenamiento del modelo YOLO.
- `src/validar.py`: validacion de metricas.
- `src/detectar.py`: deteccion en imagen, video o webcam con cajas, etiquetas en espanol y panel de estado.

Los argumentos anteriores tambien aceptan sus alias en ingles, por ejemplo
`--input`, `--output`, `--model` y `--source`.

Para cambiar el Equipo de Protección Personal obligatorio edita:

```text
configs/reglas_epp.yaml
```

## Dataset

El dataset no esta descargado dentro del proyecto. Debes descargarlo manualmente
desde los enlaces de la documentacion y colocarlo en:

```text
data/raw/
```

Despues, el preprocesamiento genera el dataset limpio en:

```text
data/processed/
```
