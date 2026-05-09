from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODELO_PREDETERMINADO = "runs/detect/runs/detect/epp_train-2/weights/best.pt"


def existe_modelo_preentrenado() -> str:
    candidatos = [
        MODELO_PREDETERMINADO,
        "runs/detect/epp_train/weights/best.pt",
        "runs/detect/epp_train-2/weights/best.pt",
    ]
    for candidato in candidatos:
        if Path(candidato).exists():
            return candidato
    return MODELO_PREDETERMINADO


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Menu principal del proyecto de deteccion de Equipo de Proteccion Personal."
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    preprocesar = subcomandos.add_parser("preprocesar", help="Limpia y prepara el dataset.")
    preprocesar.add_argument("--entrada", default="data/raw/huggingface_ppe_extraido")
    preprocesar.add_argument("--salida", default="data/processed")
    preprocesar.add_argument("--limpiar-ruido", action="store_true")
    preprocesar.add_argument("--sobrescribir", action="store_true")

    entrenar = subcomandos.add_parser("entrenar", help="Entrena el modelo YOLO.")
    entrenar.add_argument("--datos", default="data/processed/epp.yaml")
    entrenar.add_argument("--modelo", default="yolo11n.pt")
    entrenar.add_argument("--epocas", type=int, default=3)
    entrenar.add_argument("--tamano", type=int, default=416)
    entrenar.add_argument("--lote", type=int, default=4)
    entrenar.add_argument("--dispositivo", default="cpu")
    entrenar.add_argument("--nombre", default="epp_train")

    validar = subcomandos.add_parser("validar", help="Valida el modelo entrenado.")
    validar.add_argument("--modelo", default=existe_modelo_preentrenado())
    validar.add_argument("--datos", default="data/processed/epp.yaml")
    validar.add_argument("--tamano", type=int, default=640)
    validar.add_argument("--dispositivo", default="cpu")

    detectar = subcomandos.add_parser("detectar", help="Detecta en imagen, video o carpeta.")
    detectar.add_argument("--modelo", default=existe_modelo_preentrenado())
    detectar.add_argument("--fuente", required=True)
    detectar.add_argument("--salida", default="outputs")
    detectar.add_argument("--guardar", action="store_true")
    detectar.add_argument("--mostrar", action="store_true")
    detectar.add_argument("--dispositivo", default="cpu")

    camara = subcomandos.add_parser("camara", help="Abre la webcam y muestra la vista.")
    camara.add_argument("--modelo", default=existe_modelo_preentrenado())
    camara.add_argument("--fuente", default="0")
    camara.add_argument("--dispositivo", default="cpu")

    return parser.parse_args()


def ejecutar_con_argumentos(modulo: str, argumentos: list[str]) -> None:
    sys.argv = [f"python -m src.{modulo}", *argumentos]
    if modulo == "preprocesar":
        from src.preprocesar import main as ejecutar_preprocesamiento

        ejecutar_preprocesamiento()
    elif modulo == "entrenar":
        from src.entrenar import main as ejecutar_entrenamiento

        ejecutar_entrenamiento()
    elif modulo == "validar":
        from src.validar import main as ejecutar_validacion

        ejecutar_validacion()
    elif modulo == "detectar":
        from src.detectar import main as ejecutar_deteccion

        ejecutar_deteccion()
    else:
        raise ValueError(f"Modulo no soportado: {modulo}")


def main() -> None:
    argumentos = leer_argumentos()

    if argumentos.comando == "preprocesar":
        opciones = []
        if argumentos.limpiar_ruido:
            opciones.append("--limpiar-ruido")
        if argumentos.sobrescribir:
            opciones.append("--sobrescribir")
        ejecutar_con_argumentos(
            "preprocesar",
            [
                "--entrada",
                argumentos.entrada,
                "--salida",
                argumentos.salida,
                *opciones,
            ],
        )
    elif argumentos.comando == "entrenar":
        ejecutar_con_argumentos(
            "entrenar",
            [
                "--datos",
                argumentos.datos,
                "--modelo",
                argumentos.modelo,
                "--epocas",
                str(argumentos.epocas),
                "--tamano",
                str(argumentos.tamano),
                "--lote",
                str(argumentos.lote),
                "--dispositivo",
                argumentos.dispositivo,
                "--nombre",
                argumentos.nombre,
            ],
        )
    elif argumentos.comando == "validar":
        ejecutar_con_argumentos(
            "validar",
            [
                "--modelo",
                argumentos.modelo,
                "--datos",
                argumentos.datos,
                "--tamano",
                str(argumentos.tamano),
                "--dispositivo",
                argumentos.dispositivo,
            ],
        )
    elif argumentos.comando == "detectar":
        opciones = []
        if argumentos.guardar:
            opciones.append("--guardar")
        if argumentos.mostrar:
            opciones.append("--mostrar")
        ejecutar_con_argumentos(
            "detectar",
            [
                "--modelo",
                argumentos.modelo,
                "--fuente",
                argumentos.fuente,
                "--salida",
                argumentos.salida,
                "--dispositivo",
                argumentos.dispositivo,
                *opciones,
            ],
        )
    elif argumentos.comando == "camara":
        ejecutar_con_argumentos(
            "detectar",
            [
                "--modelo",
                argumentos.modelo,
                "--fuente",
                argumentos.fuente,
                "--dispositivo",
                argumentos.dispositivo,
            ],
        )


if __name__ == "__main__":
    main()
