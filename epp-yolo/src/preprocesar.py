from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm
import yaml

from .configuracion import (
    ConfiguracionEPP,
    cargar_configuracion,
    escribir_yaml_dataset,
    normalizar_nombre_clase,
)


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIVISIONES_DATASET = ("train", "valid", "val", "test")


@dataclass(frozen=True)
class ContextoDataset:
    carpeta_raiz: Path
    nombres_clases: dict[int, str]


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia imagenes y normaliza etiquetas YOLO para deteccion de Equipo de Protección Personal."
    )
    parser.add_argument(
        "--entrada",
        "--input",
        dest="entrada",
        required=True,
        help="Carpeta con datasets descargados.",
    )
    parser.add_argument(
        "--salida",
        "--output",
        dest="salida",
        required=True,
        help="Carpeta de salida procesada.",
    )
    parser.add_argument(
        "--configuracion",
        "--config",
        dest="configuracion",
        default=None,
        help="Ruta al YAML de reglas. Por defecto usa configs/reglas_epp.yaml.",
    )
    parser.add_argument(
        "--tamano",
        "--imgsz",
        dest="tamano",
        type=int,
        default=640,
        help="Tamano maximo de imagen.",
    )
    parser.add_argument(
        "--limpiar-ruido",
        "--denoise",
        dest="limpiar_ruido",
        action="store_true",
        help="Aplica reduccion de ruido mas fuerte con fastNlMeansDenoisingColored.",
    )
    parser.add_argument(
        "--sobrescribir",
        "--overwrite",
        dest="sobrescribir",
        action="store_true",
        help="Elimina la salida antes de generar el dataset procesado.",
    )
    return parser.parse_args()


def cargar_nombres_dataset(carpeta_dataset: Path) -> dict[int, str]:
    for nombre_yaml in ("data.yaml", "dataset.yaml", "epp.yaml"):
        ruta_yaml = carpeta_dataset / nombre_yaml
        if not ruta_yaml.exists():
            continue
        with ruta_yaml.open("r", encoding="utf-8") as archivo:
            datos = yaml.safe_load(archivo) or {}
        nombres = datos.get("names", {})
        if isinstance(nombres, list):
            return {indice: str(nombre) for indice, nombre in enumerate(nombres)}
        if isinstance(nombres, dict):
            return {int(indice): str(nombre) for indice, nombre in nombres.items()}
    return {}


def descubrir_contextos(carpeta_entrada: Path) -> list[ContextoDataset]:
    def parece_dataset_yolo(carpeta: Path) -> bool:
        return any((carpeta / division).exists() for division in DIVISIONES_DATASET) or (
            carpeta / "images"
        ).exists()

    if parece_dataset_yolo(carpeta_entrada):
        return [
            ContextoDataset(
                carpeta_raiz=carpeta_entrada,
                nombres_clases=cargar_nombres_dataset(carpeta_entrada),
            )
        ]

    contextos: list[ContextoDataset] = []
    for carpeta in (ruta for ruta in carpeta_entrada.iterdir() if ruta.is_dir()):
        if parece_dataset_yolo(carpeta):
            contextos.append(
                ContextoDataset(
                    carpeta_raiz=carpeta,
                    nombres_clases=cargar_nombres_dataset(carpeta),
                )
            )
    return contextos or [
        ContextoDataset(
            carpeta_raiz=carpeta_entrada,
            nombres_clases=cargar_nombres_dataset(carpeta_entrada),
        )
    ]


def inferir_division(ruta_imagen: Path, carpeta_dataset: Path) -> str:
    partes = {parte.lower() for parte in ruta_imagen.relative_to(carpeta_dataset).parts}
    if "train" in partes:
        return "train"
    if "valid" in partes or "val" in partes:
        return "valid"
    if "test" in partes:
        return "test"
    return "train"


def buscar_etiqueta(ruta_imagen: Path, carpeta_dataset: Path) -> Path | None:
    candidatas = [
        ruta_imagen.with_suffix(".txt"),
        Path(str(ruta_imagen).replace("/images/", "/labels/")).with_suffix(".txt"),
    ]
    try:
        relativa = ruta_imagen.relative_to(carpeta_dataset)
        if "images" in relativa.parts:
            partes = list(relativa.parts)
            partes[partes.index("images")] = "labels"
            candidatas.append((carpeta_dataset / Path(*partes)).with_suffix(".txt"))
    except ValueError:
        pass

    for candidata in candidatas:
        if candidata.exists():
            return candidata
    return None


def redimensionar_manteniendo_aspecto(imagen, tamano_maximo: int):
    alto, ancho = imagen.shape[:2]
    escala = min(tamano_maximo / max(alto, ancho), 1.0)
    if escala == 1.0:
        return imagen
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)
    return cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def limpiar_imagen(imagen, tamano_maximo: int, limpiar_ruido: bool):
    imagen_redimensionada = redimensionar_manteniendo_aspecto(imagen, tamano_maximo)
    if limpiar_ruido:
        imagen_redimensionada = cv2.fastNlMeansDenoisingColored(
            imagen_redimensionada, None, 5, 5, 7, 21
        )
    else:
        imagen_redimensionada = cv2.medianBlur(imagen_redimensionada, 3)

    lab = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2LAB)
    luminosidad, canal_a, canal_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagen_mejorada = cv2.merge((clahe.apply(luminosidad), canal_a, canal_b))
    return cv2.cvtColor(imagen_mejorada, cv2.COLOR_LAB2BGR)


def normalizar_archivo_etiqueta(
    ruta_etiqueta: Path,
    etiqueta_salida: Path,
    nombres_dataset: dict[int, str],
    configuracion: ConfiguracionEPP,
) -> bool:
    nombre_a_id = {nombre: indice for indice, nombre in configuracion.nombres.items()}
    lineas: list[str] = []

    for linea_cruda in ruta_etiqueta.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        valores = linea_cruda.strip().split()
        if len(valores) < 5:
            continue
        try:
            id_origen = int(float(valores[0]))
            caja = [float(valor) for valor in valores[1:5]]
        except ValueError:
            continue
        if any(valor < 0 or valor > 1 for valor in caja):
            continue

        nombre_origen = nombres_dataset.get(
            id_origen, configuracion.nombres.get(id_origen, str(id_origen))
        )
        nombre_canonico = normalizar_nombre_clase(nombre_origen, configuracion)
        if nombre_canonico not in nombre_a_id:
            continue

        id_destino = nombre_a_id[nombre_canonico]
        lineas.append(f"{id_destino} {' '.join(f'{valor:.6f}' for valor in caja)}")

    etiqueta_salida.parent.mkdir(parents=True, exist_ok=True)
    etiqueta_salida.write_text(
        "\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8"
    )
    return bool(lineas)


def listar_imagenes(contextos: list[ContextoDataset]) -> list[tuple[Path, ContextoDataset]]:
    imagenes: list[tuple[Path, ContextoDataset]] = []
    for contexto in contextos:
        for ruta in contexto.carpeta_raiz.rglob("*"):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                imagenes.append((ruta, contexto))
    return imagenes


def preprocesar_dataset(
    carpeta_entrada: Path,
    carpeta_salida: Path,
    configuracion: ConfiguracionEPP,
    tamano_imagen: int,
    limpiar_ruido: bool,
    sobrescribir: bool,
) -> None:
    if sobrescribir and carpeta_salida.exists():
        shutil.rmtree(carpeta_salida)

    for division in ("train", "valid", "test"):
        (carpeta_salida / division / "images").mkdir(parents=True, exist_ok=True)
        (carpeta_salida / division / "labels").mkdir(parents=True, exist_ok=True)

    contextos = descubrir_contextos(carpeta_entrada)
    imagenes = listar_imagenes(contextos)
    estadisticas = {"imagenes": 0, "etiquetas": 0, "omitidas": 0}

    for ruta_imagen, contexto in tqdm(imagenes, desc="Procesando imagenes"):
        ruta_etiqueta = buscar_etiqueta(ruta_imagen, contexto.carpeta_raiz)
        if ruta_etiqueta is None:
            estadisticas["omitidas"] += 1
            continue

        imagen = cv2.imread(str(ruta_imagen))
        if imagen is None:
            estadisticas["omitidas"] += 1
            continue

        division = inferir_division(ruta_imagen, contexto.carpeta_raiz)
        nombre_relativo = "_".join(
            ruta_imagen.relative_to(contexto.carpeta_raiz).with_suffix("").parts
        )
        imagen_salida = carpeta_salida / division / "images" / f"{nombre_relativo}.jpg"
        etiqueta_salida = carpeta_salida / division / "labels" / f"{nombre_relativo}.txt"

        imagen_limpia = limpiar_imagen(imagen, tamano_imagen, limpiar_ruido)
        cv2.imwrite(str(imagen_salida), imagen_limpia, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        tiene_etiquetas = normalizar_archivo_etiqueta(
            ruta_etiqueta,
            etiqueta_salida,
            contexto.nombres_clases,
            configuracion,
        )

        estadisticas["imagenes"] += 1
        estadisticas["etiquetas"] += int(tiene_etiquetas)

    escribir_yaml_dataset(carpeta_salida, configuracion)
    print(
        "Preprocesamiento terminado: "
        f"{estadisticas['imagenes']} imagenes, "
        f"{estadisticas['etiquetas']} etiquetas utiles, "
        f"{estadisticas['omitidas']} omitidas."
    )
    print(f"Dataset YOLO generado en: {carpeta_salida / 'epp.yaml'}")


def main() -> None:
    argumentos = leer_argumentos()
    configuracion = cargar_configuracion(argumentos.configuracion)
    preprocesar_dataset(
        carpeta_entrada=Path(argumentos.entrada),
        carpeta_salida=Path(argumentos.salida),
        configuracion=configuracion,
        tamano_imagen=argumentos.tamano,
        limpiar_ruido=argumentos.limpiar_ruido,
        sobrescribir=argumentos.sobrescribir,
    )


if __name__ == "__main__":
    main()
