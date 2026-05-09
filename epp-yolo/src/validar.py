from __future__ import annotations

import argparse
import json
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
        default=416,
        help="Tamano de validacion.",
    )
    parser.add_argument(
        "--dispositivo",
        "--device",
        dest="dispositivo",
        default="cpu",
        help="cpu, 0, 0,1, etc.",
    )
    parser.add_argument(
        "--salida",
        default="outputs/evaluacion_modelo.json",
        help="Archivo JSON donde se guardan las metricas.",
    )
    return parser.parse_args()


def obtener_numero(valores, indice: int) -> float | None:
    if valores is None or len(valores) <= indice:
        return None
    return float(valores[indice])


def extraer_metricas_por_clase(metricas) -> list[dict]:
    nombres = getattr(metricas, "names", {}) or {}
    indices_clases = getattr(metricas, "ap_class_index", None)
    if indices_clases is None:
        indices_clases = []
    indices_clases = list(indices_clases)
    reporte_por_clase = []

    for posicion, indice_clase in enumerate(indices_clases):
        indice_clase = int(indice_clase)
        map50_95 = obtener_numero(getattr(metricas.box, "ap", None), posicion)
        map50 = obtener_numero(getattr(metricas.box, "ap50", None), posicion)
        precision = obtener_numero(getattr(metricas.box, "p", None), posicion)
        recall = obtener_numero(getattr(metricas.box, "r", None), posicion)
        error_aproximado = None if map50_95 is None else 1.0 - map50_95

        reporte_por_clase.append(
            {
                "clase": nombres.get(indice_clase, str(indice_clase)),
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
                "mAP50_95": map50_95,
                "error_aproximado_1_menos_mAP50_95": error_aproximado,
            }
        )

    return reporte_por_clase


def construir_reporte(metricas, ruta_modelo: Path, ruta_datos: Path) -> dict:
    map_50_95 = float(metricas.box.map)
    map_50 = float(metricas.box.map50)
    map_75 = float(metricas.box.map75)
    precision_media = float(metricas.box.mp)
    recall_medio = float(metricas.box.mr)
    metricas_por_clase = extraer_metricas_por_clase(metricas)
    clases_con_error = [
        clase
        for clase in metricas_por_clase
        if clase["error_aproximado_1_menos_mAP50_95"] is not None
    ]
    clase_con_mayor_error = (
        max(clases_con_error, key=lambda clase: clase["error_aproximado_1_menos_mAP50_95"])
        if clases_con_error
        else None
    )
    return {
        "modelo": str(ruta_modelo),
        "dataset": str(ruta_datos),
        "metricas": {
            "precision_media": precision_media,
            "recall_medio": recall_medio,
            "mAP50": map_50,
            "mAP75": map_75,
            "mAP50_95": map_50_95,
        },
        "error_aproximado": {
            "1_menos_mAP50": 1.0 - map_50,
            "1_menos_mAP50_95": 1.0 - map_50_95,
            "1_menos_recall": 1.0 - recall_medio,
        },
        "metricas_por_clase": metricas_por_clase,
        "clase_con_mayor_error_aproximado": clase_con_mayor_error,
        "interpretacion": {
            "mAP50": "Mientras mas cerca de 1.0, mejor detecta objetos con IoU 0.50.",
            "mAP50_95": "Metrica mas estricta; resume precision en varios umbrales IoU.",
            "error_aproximado": "No es error clasico de regresion; se reporta como 1 - metrica.",
        },
    }


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
    reporte = construir_reporte(metricas, ruta_modelo, ruta_datos)
    ruta_salida = Path(argumentos.salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Metricas principales:")
    print(f"mAP50-95: {metricas.box.map:.4f}")
    print(f"mAP50:    {metricas.box.map50:.4f}")
    print(f"mAP75:    {metricas.box.map75:.4f}")
    print(f"Precision media: {metricas.box.mp:.4f}")
    print(f"Recall medio:    {metricas.box.mr:.4f}")
    print("Error aproximado:")
    print(f"1 - mAP50:    {1.0 - float(metricas.box.map50):.4f}")
    print(f"1 - mAP50-95: {1.0 - float(metricas.box.map):.4f}")
    print(f"1 - recall:   {1.0 - float(metricas.box.mr):.4f}")
    peor_clase = reporte["clase_con_mayor_error_aproximado"]
    if peor_clase:
        print("Clase con mayor error aproximado:")
        print(
            f"{peor_clase['clase']}: "
            f"{peor_clase['error_aproximado_1_menos_mAP50_95']:.4f}"
        )
    print(f"Reporte guardado en: {ruta_salida}")


if __name__ == "__main__":
    main()
