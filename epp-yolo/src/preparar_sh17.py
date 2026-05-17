from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from .configuracion import cargar_configuracion, escribir_yaml_dataset
from .preprocesar import normalizar_archivo_etiqueta


CLASES_SH17 = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara SH17 descargado desde Kaggle en estructura YOLO train/valid."
    )
    parser.add_argument(
        "--origen",
        default="data/raw/sh17/original",
        help="Carpeta extraida con images, labels, train_files.txt y val_files.txt.",
    )
    parser.add_argument(
        "--salida",
        default="data/raw/sh17/yolo",
        help="Carpeta de salida con estructura YOLO.",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Recrea la carpeta de salida si ya existe.",
    )
    parser.add_argument(
        "--copiar",
        action="store_true",
        help="Copia archivos en lugar de crear enlaces simbolicos.",
    )
    parser.add_argument(
        "--salida-procesada",
        default="data/processed/sh17",
        help="Carpeta procesada con clases normalizadas del proyecto.",
    )
    parser.add_argument(
        "--sin-procesada",
        action="store_true",
        help="Solo prepara data/raw/sh17/yolo y no genera data/processed/sh17.",
    )
    parser.add_argument(
        "--configuracion",
        default=None,
        help="Ruta al YAML de reglas. Por defecto usa configs/reglas_epp.yaml.",
    )
    return parser.parse_args()


def leer_split(ruta_split: Path) -> list[str]:
    if not ruta_split.exists():
        raise FileNotFoundError(f"No existe el archivo de split: {ruta_split}")
    return [
        linea.strip()
        for linea in ruta_split.read_text(encoding="utf-8", errors="ignore").splitlines()
        if linea.strip()
    ]


def enlazar_o_copiar(origen: Path, destino: Path, copiar: bool) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() or destino.is_symlink():
        destino.unlink()
    if copiar:
        shutil.copy2(origen, destino)
        return
    ruta_relativa = os.path.relpath(origen.resolve(), destino.parent.resolve())
    os.symlink(ruta_relativa, destino)


def preparar_split(
    nombres_imagenes: list[str],
    nombre_split: str,
    carpeta_origen: Path,
    carpeta_salida: Path,
    copiar: bool,
) -> tuple[int, int]:
    imagenes_origen = carpeta_origen / "images"
    etiquetas_origen = carpeta_origen / "labels"
    imagenes_salida = carpeta_salida / nombre_split / "images"
    etiquetas_salida = carpeta_salida / nombre_split / "labels"
    procesadas = 0
    omitidas = 0

    for nombre_imagen in nombres_imagenes:
        ruta_imagen = imagenes_origen / nombre_imagen
        ruta_etiqueta = etiquetas_origen / f"{Path(nombre_imagen).stem}.txt"
        if not ruta_imagen.exists() or not ruta_etiqueta.exists():
            omitidas += 1
            continue

        enlazar_o_copiar(ruta_imagen, imagenes_salida / ruta_imagen.name, copiar)
        enlazar_o_copiar(ruta_etiqueta, etiquetas_salida / ruta_etiqueta.name, copiar)
        procesadas += 1

    return procesadas, omitidas


def escribir_yaml(carpeta_salida: Path) -> Path:
    ruta_yaml = carpeta_salida / "data.yaml"
    datos = {
        "path": str(carpeta_salida.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": CLASES_SH17,
    }
    ruta_yaml.write_text(
        yaml.safe_dump(datos, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return ruta_yaml


def preparar_dataset_procesado(
    carpeta_yolo: Path,
    carpeta_procesada: Path,
    configuracion,
    copiar: bool,
    sobrescribir: bool,
) -> tuple[int, int, Path]:
    if sobrescribir and carpeta_procesada.exists():
        shutil.rmtree(carpeta_procesada)

    total_imagenes = 0
    etiquetas_utiles = 0
    for split in ("train", "valid"):
        imagenes = carpeta_yolo / split / "images"
        etiquetas = carpeta_yolo / split / "labels"
        imagenes_salida = carpeta_procesada / split / "images"
        etiquetas_salida = carpeta_procesada / split / "labels"
        imagenes_salida.mkdir(parents=True, exist_ok=True)
        etiquetas_salida.mkdir(parents=True, exist_ok=True)

        for ruta_imagen in sorted(imagenes.iterdir()):
            if not ruta_imagen.is_file():
                continue
            ruta_etiqueta = etiquetas / f"{ruta_imagen.stem}.txt"
            if not ruta_etiqueta.exists():
                continue

            enlazar_o_copiar(ruta_imagen, imagenes_salida / ruta_imagen.name, copiar)
            tiene_etiquetas = normalizar_archivo_etiqueta(
                ruta_etiqueta,
                etiquetas_salida / ruta_etiqueta.name,
                CLASES_SH17,
                configuracion,
            )
            total_imagenes += 1
            etiquetas_utiles += int(tiene_etiquetas)

    (carpeta_procesada / "test" / "images").mkdir(parents=True, exist_ok=True)
    (carpeta_procesada / "test" / "labels").mkdir(parents=True, exist_ok=True)
    ruta_yaml = escribir_yaml_dataset(carpeta_procesada, configuracion)
    return total_imagenes, etiquetas_utiles, ruta_yaml


def main() -> None:
    argumentos = leer_argumentos()
    carpeta_origen = Path(argumentos.origen)
    carpeta_salida = Path(argumentos.salida)
    configuracion = cargar_configuracion(argumentos.configuracion)

    if not carpeta_origen.exists():
        raise FileNotFoundError(f"No existe la carpeta SH17 extraida: {carpeta_origen}")
    if argumentos.sobrescribir and carpeta_salida.exists():
        shutil.rmtree(carpeta_salida)

    entrenamiento = leer_split(carpeta_origen / "train_files.txt")
    validacion = leer_split(carpeta_origen / "val_files.txt")

    train_ok, train_omitidas = preparar_split(
        entrenamiento, "train", carpeta_origen, carpeta_salida, argumentos.copiar
    )
    valid_ok, valid_omitidas = preparar_split(
        validacion, "valid", carpeta_origen, carpeta_salida, argumentos.copiar
    )
    ruta_yaml = escribir_yaml(carpeta_salida)

    print("SH17 preparado para YOLO.")
    print(f"Train: {train_ok} imagenes, {train_omitidas} omitidas.")
    print(f"Valid: {valid_ok} imagenes, {valid_omitidas} omitidas.")
    print(f"YAML generado en: {ruta_yaml}")

    if not argumentos.sin_procesada:
        total, utiles, ruta_yaml_procesado = preparar_dataset_procesado(
            carpeta_salida,
            Path(argumentos.salida_procesada),
            configuracion,
            argumentos.copiar,
            argumentos.sobrescribir,
        )
        print("Dataset SH17 normalizado para el proyecto.")
        print(f"Imagenes procesadas: {total}")
        print(f"Archivos con etiquetas utiles: {utiles}")
        print(f"YAML procesado generado en: {ruta_yaml_procesado}")

    print("Entrenamiento recomendado:")
    print("python main.py entrenar --datos data/processed/sh17/epp.yaml --epocas 30 --tamano 640 --lote 8")


if __name__ == "__main__":
    main()
