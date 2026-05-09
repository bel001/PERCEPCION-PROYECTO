from __future__ import annotations

import argparse
from pathlib import Path


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida un modelo YOLO entrenado.")
    parser.add_argument(
        "--modelo",
        "--model",
        dest="modelo",
        required=True,
        help="Ruta a best.pt o pesos YOLO.",
    )
    parser.add_argument(
        "--datos",
        "--data",
        dest="datos",
        required=True,
        help="Ruta a epp.yaml.",
    )
    parser.add_argument(
        "--tamano",
        "--imgsz",
        dest="tamano",
        type=int,
        default=640,
        help="Tamano de validacion.",
    )
    parser.add_argument(
        "--dispositivo",
        "--device",
        dest="dispositivo",
        default="cpu",
        help="cpu, 0, 0,1, etc.",
    )
    return parser.parse_args()


def main() -> None:
    argumentos = leer_argumentos()
    ruta_modelo = Path(argumentos.modelo)
    ruta_datos = Path(argumentos.datos)
    if not ruta_modelo.exists() and not str(ruta_modelo).startswith("yolo"):
        raise FileNotFoundError(f"No existe el modelo: {ruta_modelo}")
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No existe el dataset YAML: {ruta_datos}")

    from ultralytics import YOLO

    modelo = YOLO(str(ruta_modelo))
    metricas = modelo.val(
        data=str(ruta_datos), imgsz=argumentos.tamano, device=argumentos.dispositivo
    )
    print("Metricas principales:")
    print(f"mAP50-95: {metricas.box.map:.4f}")
    print(f"mAP50:    {metricas.box.map50:.4f}")
    print(f"mAP75:    {metricas.box.map75:.4f}")


if __name__ == "__main__":
    main()
