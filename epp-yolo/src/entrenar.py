from __future__ import annotations

import argparse
from pathlib import Path


def resolver_yaml_dataset(ruta_datos: Path) -> Path:
    if ruta_datos.is_file():
        return ruta_datos
    if ruta_datos.is_dir():
        for nombre in ("epp.yaml", "data.yaml", "dataset.yaml", "dataset_auto.yaml"):
            candidato = ruta_datos / nombre
            if candidato.exists():
                return candidato
    raise FileNotFoundError(
        f"No existe el archivo de datos: {ruta_datos}\n"
        "Ejecuta primero: python main.py revisar-dataset --ruta data"
    )


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena YOLO para detectar Equipo de Protección Personal."
    )
    parser.add_argument(
        "--datos",
        "--data",
        dest="datos",
        required=True,
        help="Ruta a data.yaml/epp.yaml.",
    )
    parser.add_argument(
        "--modelo",
        "--model",
        dest="modelo",
        default="yolo11n.pt",
        help="Pesos o modelo base YOLO.",
    )
    parser.add_argument(
        "--epocas",
        "--epochs",
        dest="epocas",
        type=int,
        default=50,
        help="Numero de epocas.",
    )
    parser.add_argument(
        "--tamano",
        "--imgsz",
        dest="tamano",
        type=int,
        default=640,
        help="Tamano de entrenamiento.",
    )
    parser.add_argument(
        "--lote",
        "--batch",
        dest="lote",
        type=int,
        default=8,
        help="Tamano de batch.",
    )
    parser.add_argument(
        "--dispositivo",
        "--device",
        dest="dispositivo",
        default="cpu",
        help="cpu, 0, 0,1, etc.",
    )
    parser.add_argument(
        "--proyecto",
        "--project",
        dest="proyecto",
        default="runs/detect",
        help="Carpeta de resultados.",
    )
    parser.add_argument(
        "--nombre",
        "--name",
        dest="nombre",
        default="epp_sh17",
        help="Nombre del experimento.",
    )
    return parser.parse_args()


def main() -> None:
    argumentos = leer_argumentos()
    ruta_datos = resolver_yaml_dataset(Path(argumentos.datos))
    carpeta_proyecto = Path(argumentos.proyecto)
    if not carpeta_proyecto.is_absolute():
        carpeta_proyecto = Path.cwd() / carpeta_proyecto

    from ultralytics import YOLO

    modelo = YOLO(argumentos.modelo)
    resultados = modelo.train(
        data=str(ruta_datos),
        epochs=argumentos.epocas,
        imgsz=argumentos.tamano,
        batch=argumentos.lote,
        device=argumentos.dispositivo,
        project=str(carpeta_proyecto),
        name=argumentos.nombre,
    )
    print(f"Entrenamiento finalizado. Resultados: {resultados.save_dir}")


if __name__ == "__main__":
    main()
