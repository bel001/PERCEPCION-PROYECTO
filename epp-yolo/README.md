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
python main.py evaluar
python main.py camara
```

Si la camara va lenta en CPU:

```bash
python main.py camara --tamano 320 --saltar-frames 3 --ancho-camara 640 --alto-camara 480 --ancho-ventana 800 --alto-ventana 600
```

Si la ventana se ve cortada:

```bash
python main.py camara --ancho-ventana 800 --alto-ventana 600
```

Si la camara sale de lado:

```bash
python main.py camara --rotar 90
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
python main.py evaluar
python main.py detectar --fuente ruta/imagen.jpg
python main.py camara
```

Si existe `.venv`, `main.py` intenta relanzarse automaticamente con el Python
del entorno virtual. Aun asi, la forma recomendada es activar el entorno:

```bash
source .venv/bin/activate
python main.py camara
```

Tambien puedes llamar directamente los modulos, por ejemplo
`python -m src.detectar --modelo runs/detect/runs/detect/epp_train-2/weights/best.pt --fuente 0`.

## Capas principales

- `main.py`: punto de entrada principal para ejecutar todo con comandos simples.
- `src/preprocesar.py`: limpieza de ruido, mejora de contraste y normalizacion de etiquetas.
- `src/entrenar.py`: entrenamiento del modelo YOLO.
- `src/validar.py`: validacion de metricas y reporte de error aproximado.
- `src/detectar.py`: deteccion en imagen, video o webcam con cajas, etiquetas en espanol y panel de estado.

## Evaluacion del modelo

```bash
python main.py evaluar --tamano 416 --salida outputs/evaluacion_modelo.json
cat outputs/evaluacion_modelo.json
```

El reporte incluye `precision_media`, `recall_medio`, `mAP50`, `mAP50_95` y
errores aproximados como `1 - mAP50`. Tambien incluye `metricas_por_clase` y
`clase_con_mayor_error_aproximado` para saber donde falla mas el modelo.

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
