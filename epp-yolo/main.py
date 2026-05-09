from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parent
PYTHON_VENV = RAIZ_PROYECTO / ".venv" / "bin" / "python"
MODELO_PREDETERMINADO = "runs/detect/runs/detect/epp_train-2/weights/best.pt"


def relanzar_con_venv_si_existe() -> None:
    if not PYTHON_VENV.exists():
        return
    if Path(sys.prefix).resolve() == (RAIZ_PROYECTO / ".venv").resolve():
        return
    python_venv = PYTHON_VENV
    os.execv(str(python_venv), [str(python_venv), str(Path(__file__).resolve()), *sys.argv[1:]])


def usar_raiz_del_proyecto() -> None:
    os.chdir(RAIZ_PROYECTO)


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
    validar.add_argument("--tamano", type=int, default=416)
    validar.add_argument("--dispositivo", default="cpu")
    validar.add_argument("--salida", default="outputs/evaluacion_modelo.json")

    evaluar = subcomandos.add_parser(
        "evaluar", help="Evalua el modelo y guarda metricas/error en JSON."
    )
    evaluar.add_argument("--modelo", default=existe_modelo_preentrenado())
    evaluar.add_argument("--datos", default="data/processed/epp.yaml")
    evaluar.add_argument("--tamano", type=int, default=416)
    evaluar.add_argument("--dispositivo", default="cpu")
    evaluar.add_argument("--salida", default="outputs/evaluacion_modelo.json")

    detectar = subcomandos.add_parser("detectar", help="Detecta en imagen, video o carpeta.")
    detectar.add_argument("--modelo", default=existe_modelo_preentrenado())
    detectar.add_argument("--fuente", required=True)
    detectar.add_argument("--salida", default="outputs")
    detectar.add_argument("--guardar", action="store_true")
    detectar.add_argument("--mostrar", action="store_true")
    detectar.add_argument("--dispositivo", default="cpu")
    detectar.add_argument("--tamano", type=int, default=416)
    detectar.add_argument("--saltar-frames", type=int, default=1)

    camara = subcomandos.add_parser("camara", help="Abre la webcam y muestra la vista.")
    camara.add_argument("--modelo", default=existe_modelo_preentrenado())
    camara.add_argument("--fuente", default="0")
    camara.add_argument("--dispositivo", default="cpu")
    camara.add_argument("--tamano", type=int, default=416)
    camara.add_argument("--saltar-frames", type=int, default=2)
    camara.add_argument("--ancho-camara", type=int, default=640)
    camara.add_argument("--alto-camara", type=int, default=480)
    camara.add_argument("--ancho-ventana", type=int, default=800)
    camara.add_argument("--alto-ventana", type=int, default=600)
    camara.add_argument("--rotar", choices=("0", "90", "180", "270"), default="0")

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
    relanzar_con_venv_si_existe()
    usar_raiz_del_proyecto()
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
    elif argumentos.comando in {"validar", "evaluar"}:
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
                "--salida",
                argumentos.salida,
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
                "--tamano",
                str(argumentos.tamano),
                "--saltar-frames",
                str(argumentos.saltar_frames),
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
                "--tamano",
                str(argumentos.tamano),
                "--saltar-frames",
                str(argumentos.saltar_frames),
                "--ancho-camara",
                str(argumentos.ancho_camara),
                "--alto-camara",
                str(argumentos.alto_camara),
                "--ancho-ventana",
                str(argumentos.ancho_ventana),
                "--alto-ventana",
                str(argumentos.alto_ventana),
                "--rotar",
                argumentos.rotar,
            ],
        )


if __name__ == "__main__":
    main()
