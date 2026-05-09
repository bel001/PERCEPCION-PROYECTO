from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .configuracion import (
    ConfiguracionEPP,
    cargar_configuracion,
    normalizar_nombre_clase,
)


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
NOMBRES_VISIBLES = {
    "person": "trabajador",
    "helmet": "casco",
    "vest": "chaleco",
    "gloves": "guantes",
    "goggles": "gafas",
    "mask": "mascarilla",
    "safety_shoes": "zapatos de seguridad",
    "no_helmet": "sin casco",
    "no_vest": "sin chaleco",
}


@dataclass
class Deteccion:
    etiqueta: str
    confianza: float
    caja: tuple[int, int, int, int]


@dataclass
class EstadoTrabajador:
    persona: Deteccion
    epp_presente: set[str] = field(default_factory=set)
    epp_faltante: set[str] = field(default_factory=set)


def nombre_visible(etiqueta: str) -> str:
    return NOMBRES_VISIBLES.get(etiqueta, etiqueta.replace("_", " "))


def lista_visible(etiquetas: Iterable[str]) -> str:
    return ", ".join(nombre_visible(etiqueta) for etiqueta in sorted(etiquetas))


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta Equipo de Protección Personal en imagenes, videos o webcam y genera alertas."
    )
    parser.add_argument(
        "--modelo",
        "--model",
        dest="modelo",
        required=True,
        help="Ruta al modelo entrenado.",
    )
    parser.add_argument(
        "--fuente",
        "--source",
        dest="fuente",
        required=True,
        help="Imagen, video, carpeta o indice de webcam. Ejemplo: 0",
    )
    parser.add_argument(
        "--configuracion",
        "--config",
        dest="configuracion",
        default=None,
        help="Ruta al YAML de reglas. Por defecto usa configs/reglas_epp.yaml.",
    )
    parser.add_argument(
        "--salida",
        "--output",
        dest="salida",
        default="outputs",
        help="Carpeta de salida.",
    )
    parser.add_argument(
        "--mostrar",
        "--show",
        dest="mostrar",
        action="store_true",
        help="Muestra ventana en vivo.",
    )
    parser.add_argument(
        "--guardar",
        "--save",
        dest="guardar",
        action="store_true",
        help="Guarda imagen/video anotado.",
    )
    parser.add_argument(
        "--dispositivo",
        "--device",
        dest="dispositivo",
        default="cpu",
        help="cpu, 0, 0,1, etc.",
    )
    parser.add_argument(
        "--tamano",
        "--imgsz",
        dest="tamano",
        type=int,
        default=416,
        help="Tamano de inferencia. Menor es mas rapido; 416 recomendado para CPU.",
    )
    parser.add_argument(
        "--saltar-frames",
        dest="saltar_frames",
        type=int,
        default=1,
        help="Procesa 1 de cada N frames en video/camara. Usa 2 o 3 si va lento.",
    )
    parser.add_argument(
        "--ancho-camara",
        dest="ancho_camara",
        type=int,
        default=640,
        help="Ancho solicitado para webcam.",
    )
    parser.add_argument(
        "--alto-camara",
        dest="alto_camara",
        type=int,
        default=480,
        help="Alto solicitado para webcam.",
    )
    parser.add_argument(
        "--ancho-ventana",
        dest="ancho_ventana",
        type=int,
        default=960,
        help="Ancho maximo de la ventana mostrada.",
    )
    parser.add_argument(
        "--alto-ventana",
        dest="alto_ventana",
        type=int,
        default=600,
        help="Alto maximo de la ventana mostrada.",
    )
    parser.add_argument(
        "--rotar",
        choices=("0", "90", "180", "270"),
        default="0",
        help="Rota la imagen si la camara sale de lado.",
    )
    return parser.parse_args()


def calcular_iou(
    caja_a: tuple[int, int, int, int], caja_b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    ancho_interseccion = max(0, ix2 - ix1)
    alto_interseccion = max(0, iy2 - iy1)
    interseccion = ancho_interseccion * alto_interseccion
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - interseccion
    return interseccion / union if union else 0.0


def centro_dentro_con_margen(
    caja_interna: tuple[int, int, int, int],
    caja_externa: tuple[int, int, int, int],
    margen: float,
) -> bool:
    ix1, iy1, ix2, iy2 = caja_interna
    ox1, oy1, ox2, oy2 = caja_externa
    centro_x = (ix1 + ix2) / 2
    centro_y = (iy1 + iy2) / 2
    ancho = ox2 - ox1
    alto = oy2 - oy1
    return (
        ox1 - ancho * margen <= centro_x <= ox2 + ancho * margen
        and oy1 - alto * margen <= centro_y <= oy2 + alto * margen
    )


def detecciones_desde_resultado(resultado, configuracion: ConfiguracionEPP) -> list[Deteccion]:
    detecciones: list[Deteccion] = []
    for caja_yolo in resultado.boxes:
        confianza = float(caja_yolo.conf[0])
        if confianza < configuracion.umbrales.confianza:
            continue
        id_clase = int(caja_yolo.cls[0])
        etiqueta_cruda = resultado.names.get(id_clase, str(id_clase))
        etiqueta = normalizar_nombre_clase(etiqueta_cruda, configuracion)
        xyxy = caja_yolo.xyxy[0].cpu().numpy().astype(int).tolist()
        detecciones.append(
            Deteccion(
                etiqueta=etiqueta,
                confianza=confianza,
                caja=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            )
        )
    return detecciones


def evaluar_trabajadores(
    detecciones: Iterable[Deteccion], configuracion: ConfiguracionEPP
) -> list[EstadoTrabajador]:
    detecciones = list(detecciones)
    personas = [deteccion for deteccion in detecciones if deteccion.etiqueta == "person"]
    elementos_epp = [
        deteccion for deteccion in detecciones if deteccion.etiqueta != "person"
    ]
    estados = [EstadoTrabajador(persona=persona) for persona in personas]

    if not estados:
        return []

    for elemento in elementos_epp:
        mejor_estado = max(
            estados,
            key=lambda estado: calcular_iou(elemento.caja, estado.persona.caja),
        )
        superposicion = calcular_iou(elemento.caja, mejor_estado.persona.caja)
        centrado = centro_dentro_con_margen(
            elemento.caja,
            mejor_estado.persona.caja,
            configuracion.umbrales.margen_centro_asociacion,
        )
        if superposicion < configuracion.umbrales.iou_asociacion and not centrado:
            continue

        clase_negativa_a_epp = {
            clase_negativa: requerido
            for requerido, clase_negativa in configuracion.clases_negativas.items()
        }
        epp_negativo = clase_negativa_a_epp.get(elemento.etiqueta)
        if epp_negativo:
            mejor_estado.epp_faltante.add(epp_negativo)
        else:
            mejor_estado.epp_presente.add(elemento.etiqueta)

    for estado in estados:
        for requerido in configuracion.epp_obligatorio:
            if requerido not in estado.epp_presente:
                estado.epp_faltante.add(requerido)
    return estados


def dibujar_texto(
    fotograma,
    texto: str,
    origen: tuple[int, int],
    color: tuple[int, int, int],
    escala: float,
    grosor: int,
) -> None:
    x, y = origen
    cv2.putText(
        fotograma,
        texto,
        (x, max(20, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color,
        grosor,
        cv2.LINE_AA,
    )


def dibujar_panel_estado(
    fotograma,
    detecciones: list[Deteccion],
    estados: list[EstadoTrabajador],
    configuracion: ConfiguracionEPP,
) -> None:
    conteo_objetos = Counter(
        nombre_visible(deteccion.etiqueta)
        for deteccion in detecciones
        if deteccion.etiqueta != "person"
    )
    infracciones = [estado for estado in estados if estado.epp_faltante]

    if conteo_objetos:
        objetos = ", ".join(
            f"{nombre} x{cantidad}" for nombre, cantidad in conteo_objetos.most_common(5)
        )
    else:
        objetos = "sin objetos detectados"

    lineas = [
        "Camara observa:",
        f"Trabajadores detectados: {len(estados)}",
        f"Objetos: {objetos}",
    ]
    if not estados:
        lineas.append("Estado: no se detecta trabajador")
    elif infracciones:
        lineas.append(f"Alerta: {len(infracciones)} trabajador(es) con Equipo de Proteccion Personal incompleto")
    else:
        lineas.append("Estado: trabajadores con Equipo de Proteccion Personal completo")

    alto_panel = 24 + 26 * len(lineas)
    ancho_panel = min(fotograma.shape[1] - 20, 760)
    capa = fotograma.copy()
    cv2.rectangle(capa, (10, 10), (10 + ancho_panel, alto_panel), (25, 25, 25), -1)
    cv2.addWeighted(capa, 0.72, fotograma, 0.28, 0, fotograma)

    for indice, linea in enumerate(lineas):
        color = configuracion.alertas.color_peligro if "Alerta:" in linea else (255, 255, 255)
        dibujar_texto(
            fotograma,
            linea,
            (24, 38 + indice * 26),
            color,
            0.62,
            2,
        )


def anotar_fotograma(
    fotograma, detecciones: list[Deteccion], configuracion: ConfiguracionEPP
):
    estados = evaluar_trabajadores(detecciones, configuracion)
    ids_personas = {id(estado.persona) for estado in estados}

    for deteccion in detecciones:
        if id(deteccion) in ids_personas:
            continue
        x1, y1, x2, y2 = deteccion.caja
        cv2.rectangle(fotograma, (x1, y1), (x2, y2), configuracion.alertas.color_epp, 2)
        dibujar_texto(
            fotograma,
            f"{nombre_visible(deteccion.etiqueta)} {deteccion.confianza:.2f}",
            (x1, y1 - 8),
            configuracion.alertas.color_epp,
            configuracion.alertas.escala_texto,
            configuracion.alertas.grosor_texto,
        )

    for indice, estado in enumerate(estados, start=1):
        x1, y1, x2, y2 = estado.persona.caja
        faltantes = sorted(estado.epp_faltante)
        esta_seguro = not faltantes
        color = (
            configuracion.alertas.color_seguro
            if esta_seguro
            else configuracion.alertas.color_peligro
        )
        etiqueta = "OK" if esta_seguro else f"FALTA: {lista_visible(faltantes)}"
        cv2.rectangle(fotograma, (x1, y1), (x2, y2), color, 3)
        dibujar_texto(
            fotograma,
            f"Trabajador {indice}: {etiqueta}",
            (x1, y1 - 10),
            color,
            configuracion.alertas.escala_texto,
            configuracion.alertas.grosor_texto,
        )

    dibujar_panel_estado(fotograma, detecciones, estados, configuracion)

    if configuracion.alertas.habilitada and any(
        estado.epp_faltante for estado in estados
    ):
        dibujar_texto(
            fotograma,
            "ALERTA: incumplimiento de Equipo de Proteccion Personal",
            (20, 35),
            configuracion.alertas.color_peligro,
            0.9,
            configuracion.alertas.grosor_texto + 1,
        )
    return fotograma, estados


def predecir_fotograma(
    modelo,
    fotograma,
    configuracion: ConfiguracionEPP,
    dispositivo: str,
    tamano: int,
):
    resultados = modelo.predict(
        source=fotograma,
        conf=configuracion.umbrales.confianza,
        iou=configuracion.umbrales.iou,
        device=dispositivo,
        imgsz=tamano,
        verbose=False,
    )
    detecciones = detecciones_desde_resultado(resultados[0], configuracion)
    anotado, estados = anotar_fotograma(fotograma.copy(), detecciones, configuracion)
    return anotado, estados, detecciones


def rotar_fotograma(fotograma, rotacion: str):
    if rotacion == "90":
        return cv2.rotate(fotograma, cv2.ROTATE_90_CLOCKWISE)
    if rotacion == "180":
        return cv2.rotate(fotograma, cv2.ROTATE_180)
    if rotacion == "270":
        return cv2.rotate(fotograma, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return fotograma


def ajustar_a_ventana(fotograma, ancho_maximo: int, alto_maximo: int):
    alto, ancho = fotograma.shape[:2]
    escala = min(ancho_maximo / ancho, alto_maximo / alto)
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    redimensionado = cv2.resize(
        fotograma, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA
    )
    lienzo = np.zeros((alto_maximo, ancho_maximo, 3), dtype=redimensionado.dtype)
    x = (ancho_maximo - nuevo_ancho) // 2
    y = (alto_maximo - nuevo_alto) // 2
    lienzo[y : y + nuevo_alto, x : x + nuevo_ancho] = redimensionado
    return lienzo


def es_fuente_webcam(fuente: str) -> bool:
    return fuente.isdigit()


def procesar_imagen(
    modelo,
    fuente: Path,
    carpeta_salida: Path,
    configuracion: ConfiguracionEPP,
    guardar: bool,
    dispositivo: str,
    tamano: int,
):
    imagen = cv2.imread(str(fuente))
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {fuente}")
    anotada, estados, _ = predecir_fotograma(
        modelo, imagen, configuracion, dispositivo, tamano
    )
    if guardar:
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        ruta_salida = carpeta_salida / f"{fuente.stem}_epp.jpg"
        cv2.imwrite(str(ruta_salida), anotada)
        print(f"Imagen guardada en: {ruta_salida}")
    imprimir_resumen(estados)


def procesar_video(
    modelo,
    fuente: str | int,
    carpeta_salida: Path,
    configuracion: ConfiguracionEPP,
    guardar: bool,
    mostrar: bool,
    dispositivo: str,
    tamano: int,
    saltar_frames: int,
    ancho_camara: int | None = None,
    alto_camara: int | None = None,
    ancho_ventana: int = 960,
    alto_ventana: int = 720,
    rotacion: str = "0",
):
    captura = cv2.VideoCapture(fuente)
    if not captura.isOpened():
        raise ValueError(f"No se pudo abrir la fuente de video: {fuente}")
    if isinstance(fuente, int):
        captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho_camara or 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto_camara or 480)
        captura.set(cv2.CAP_PROP_FPS, 30)
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    escritor = None
    if guardar:
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = captura.get(cv2.CAP_PROP_FPS) or 25
        ruta_salida = carpeta_salida / "deteccion_epp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        escritor = cv2.VideoWriter(str(ruta_salida), fourcc, fps, (ancho, alto))
        print(f"Guardando video en: {ruta_salida}")

    ultimos_estados: list[EstadoTrabajador] = []
    ultimas_detecciones: list[Deteccion] = []
    contador_frames = 0
    saltar_frames = max(1, saltar_frames)
    nombre_ventana = "Deteccion de Equipo de Proteccion Personal"
    if mostrar:
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_ventana, ancho_ventana, alto_ventana)
    while True:
        correcto, fotograma = captura.read()
        if not correcto:
            break
        fotograma = rotar_fotograma(fotograma, rotacion)
        contador_frames += 1
        if (contador_frames - 1) % saltar_frames == 0 or not ultimas_detecciones:
            anotado, ultimos_estados, ultimas_detecciones = predecir_fotograma(
                modelo, fotograma, configuracion, dispositivo, tamano
            )
        else:
            anotado, ultimos_estados = anotar_fotograma(
                fotograma.copy(), ultimas_detecciones, configuracion
            )
        if escritor:
            escritor.write(anotado)
        if mostrar:
            vista = ajustar_a_ventana(anotado, ancho_ventana, alto_ventana)
            cv2.imshow(nombre_ventana, vista)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    captura.release()
    if escritor:
        escritor.release()
    if mostrar:
        cv2.destroyAllWindows()
    imprimir_resumen(ultimos_estados)


def procesar_carpeta(
    modelo,
    fuente: Path,
    carpeta_salida: Path,
    configuracion: ConfiguracionEPP,
    guardar: bool,
    dispositivo: str,
    tamano: int,
):
    imagenes = [
        ruta for ruta in fuente.rglob("*") if ruta.suffix.lower() in EXTENSIONES_IMAGEN
    ]
    if not imagenes:
        raise ValueError(f"No se encontraron imagenes en: {fuente}")
    for ruta_imagen in imagenes:
        procesar_imagen(
            modelo, ruta_imagen, carpeta_salida, configuracion, guardar, dispositivo, tamano
        )


def imprimir_resumen(estados: list[EstadoTrabajador]) -> None:
    if not estados:
        print("No se detectaron trabajadores.")
        return
    infracciones = [estado for estado in estados if estado.epp_faltante]
    print(f"Trabajadores detectados: {len(estados)}")
    print(f"Infracciones detectadas: {len(infracciones)}")
    for indice, estado in enumerate(estados, start=1):
        faltante = lista_visible(estado.epp_faltante) if estado.epp_faltante else "ninguno"
        print(f"  Trabajador {indice}: faltante={faltante}")


def main() -> None:
    argumentos = leer_argumentos()
    configuracion = cargar_configuracion(argumentos.configuracion)
    carpeta_salida = Path(argumentos.salida)

    from ultralytics import YOLO

    modelo = YOLO(argumentos.modelo)
    fuente = argumentos.fuente

    if es_fuente_webcam(fuente):
        procesar_video(
            modelo,
            int(fuente),
            carpeta_salida,
            configuracion,
            argumentos.guardar,
            True,
            argumentos.dispositivo,
            argumentos.tamano,
            argumentos.saltar_frames,
            argumentos.ancho_camara,
            argumentos.alto_camara,
            argumentos.ancho_ventana,
            argumentos.alto_ventana,
            argumentos.rotar,
        )
        return

    ruta_fuente = Path(fuente)
    if ruta_fuente.is_dir():
        procesar_carpeta(
            modelo,
            ruta_fuente,
            carpeta_salida,
            configuracion,
            True,
            argumentos.dispositivo,
            argumentos.tamano,
        )
    elif ruta_fuente.suffix.lower() in EXTENSIONES_IMAGEN:
        procesar_imagen(
            modelo,
            ruta_fuente,
            carpeta_salida,
            configuracion,
            True,
            argumentos.dispositivo,
            argumentos.tamano,
        )
    elif ruta_fuente.suffix.lower() in EXTENSIONES_VIDEO:
        procesar_video(
            modelo,
            str(ruta_fuente),
            carpeta_salida,
            configuracion,
            argumentos.guardar,
            argumentos.mostrar,
            argumentos.dispositivo,
            argumentos.tamano,
            argumentos.saltar_frames,
            ancho_ventana=argumentos.ancho_ventana,
            alto_ventana=argumentos.alto_ventana,
            rotacion=argumentos.rotar,
        )
    else:
        raise ValueError(f"Fuente no soportada: {fuente}")


if __name__ == "__main__":
    main()
