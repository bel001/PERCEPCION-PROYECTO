# PERCEPCION-PROYECTO

Sistema de vision artificial para deteccion de Equipo de Proteccion Personal
con YOLO.

## Contenido

- `epp-yolo/`: proyecto Python con scripts para preprocesar, entrenar, validar y detectar.
- `documentacion/README.md`: explicacion completa del proyecto, datasets y comandos.

## Uso rapido

```bash
cd epp-yolo
source .venv/bin/activate
python main.py --help
```

Comandos principales:

```bash
python main.py preprocesar --entrada data/raw/huggingface_ppe_extraido --salida data/processed --limpiar-ruido --sobrescribir
python main.py entrenar --epocas 3 --tamano 416 --lote 4
python main.py validar
python main.py camara
```

Los datasets, pesos entrenados, entornos virtuales y salidas generadas no se
suben al repositorio.
