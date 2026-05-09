from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CONFIGURACION_PREDETERMINADA = (
    Path(__file__).resolve().parents[1] / "configs" / "reglas_epp.yaml"
)


@dataclass(frozen=True)
class Umbrales:
    confianza: float = 0.35
    iou: float = 0.45
    iou_asociacion: float = 0.02
    margen_centro_asociacion: float = 0.15


@dataclass(frozen=True)
class EstiloAlerta:
    habilitada: bool = True
    color_seguro: tuple[int, int, int] = (0, 180, 0)
    color_peligro: tuple[int, int, int] = (0, 0, 255)
    color_epp: tuple[int, int, int] = (255, 170, 0)
    escala_texto: float = 0.65
    grosor_texto: int = 2


@dataclass(frozen=True)
class ConfiguracionEPP:
    nombres: dict[int, str]
    epp_obligatorio: list[str]
    umbrales: Umbrales
    alertas: EstiloAlerta
    alias_clases: dict[str, str]
    clases_negativas: dict[str, str]

    @property
    def nombres_canonicos(self) -> list[str]:
        return [nombre for _, nombre in sorted(self.nombres.items())]


def _como_color(valor: Any, respaldo: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(valor, list | tuple) or len(valor) != 3:
        return respaldo
    return tuple(max(0, min(255, int(canal))) for canal in valor)


def cargar_configuracion(ruta: str | Path | None = None) -> ConfiguracionEPP:
    ruta_configuracion = Path(ruta) if ruta else CONFIGURACION_PREDETERMINADA
    with ruta_configuracion.open("r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo) or {}

    nombres_crudos = datos.get("nombres", datos.get("names", {}))
    nombres = {int(indice): str(nombre) for indice, nombre in nombres_crudos.items()}

    umbrales = datos.get("umbrales", datos.get("thresholds", {}))
    estilo_alerta = datos.get("alertas", datos.get("alerts", {}))
    epp_obligatorio = datos.get("epp_obligatorio", datos.get("required_ppe", []))
    alias_clases = datos.get("alias_clases", datos.get("class_aliases", {}))
    clases_negativas = datos.get("clases_negativas", datos.get("negative_classes", {}))

    return ConfiguracionEPP(
        nombres=nombres,
        epp_obligatorio=[str(elemento) for elemento in epp_obligatorio],
        umbrales=Umbrales(
            confianza=float(umbrales.get("confianza", umbrales.get("confidence", 0.35))),
            iou=float(umbrales.get("iou", 0.45)),
            iou_asociacion=float(
                umbrales.get("iou_asociacion", umbrales.get("association_iou", 0.02))
            ),
            margen_centro_asociacion=float(
                umbrales.get(
                    "margen_centro_asociacion",
                    umbrales.get("association_center_margin", 0.15),
                )
            ),
        ),
        alertas=EstiloAlerta(
            habilitada=bool(estilo_alerta.get("habilitada", estilo_alerta.get("enabled", True))),
            color_seguro=_como_color(
                estilo_alerta.get("color_seguro", estilo_alerta.get("safe_color")),
                (0, 180, 0),
            ),
            color_peligro=_como_color(
                estilo_alerta.get("color_peligro", estilo_alerta.get("danger_color")),
                (0, 0, 255),
            ),
            color_epp=_como_color(
                estilo_alerta.get("color_epp", estilo_alerta.get("ppe_color")),
                (255, 170, 0),
            ),
            escala_texto=float(
                estilo_alerta.get("escala_texto", estilo_alerta.get("text_scale", 0.65))
            ),
            grosor_texto=int(
                estilo_alerta.get("grosor_texto", estilo_alerta.get("text_thickness", 2))
            ),
        ),
        alias_clases={
            _normalizar_clave(str(alias)): _normalizar_clave(str(destino))
            for alias, destino in alias_clases.items()
        },
        clases_negativas={
            _normalizar_clave(str(epp)): _normalizar_clave(str(clase_negativa))
            for epp, clase_negativa in clases_negativas.items()
        },
    )


def _normalizar_clave(nombre: str) -> str:
    return "_".join(nombre.strip().lower().replace("-", "_").split())


def normalizar_nombre_clase(nombre: str, configuracion: ConfiguracionEPP) -> str:
    nombre_normalizado = _normalizar_clave(nombre)
    return configuracion.alias_clases.get(nombre_normalizado, nombre_normalizado)


def escribir_yaml_dataset(
    carpeta_salida: Path,
    configuracion: ConfiguracionEPP,
    entrenamiento: str = "train/images",
    validacion: str = "valid/images",
    prueba: str = "test/images",
) -> Path:
    ruta_yaml = carpeta_salida / "epp.yaml"
    contenido = {
        "path": str(carpeta_salida.resolve()),
        "train": entrenamiento,
        "val": validacion,
        "test": prueba,
        "names": configuracion.nombres,
    }
    with ruta_yaml.open("w", encoding="utf-8") as archivo:
        yaml.safe_dump(contenido, archivo, sort_keys=False, allow_unicode=True)
    return ruta_yaml
