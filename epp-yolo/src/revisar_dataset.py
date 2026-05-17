from __future__ import annotations

import argparse
from pathlib import Path

import yaml


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTENSIONES_YAML = {".yaml", ".yml"}
NOMBRES_YAML_PRIORITARIOS = ("epp.yaml", "data.yaml", "dataset.yaml", "dataset_auto.yaml")


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revisa si una carpeta esta lista para entrenar YOLO."
    )
    parser.add_argument(
        "--ruta",
        default="data",
        help="Carpeta donde se importo el dataset.",
    )
    parser.add_argument(
        "--crear-yaml",
        action="store_true",
        help="Crea data/dataset_auto.yaml si encuentra estructura YOLO valida.",
    )
    parser.add_argument(
        "--salida",
        default="data/dataset_auto.yaml",
        help="Ruta del YAML generado con --crear-yaml.",
    )
    return parser.parse_args()


def buscar_yaml(carpeta: Path) -> list[Path]:
    archivos = [ruta for ruta in carpeta.rglob("*") if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_YAML]
    return sorted(
        archivos,
        key=lambda ruta: (
            ruta.name not in NOMBRES_YAML_PRIORITARIOS,
            len(ruta.parts),
            str(ruta),
        ),
    )


def cargar_yaml(ruta_yaml: Path) -> dict:
    with ruta_yaml.open("r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo) or {}
    if not isinstance(datos, dict):
        raise ValueError(f"El YAML no contiene un diccionario valido: {ruta_yaml}")
    return datos


def normalizar_nombres(nombres) -> dict[int, str]:
    if isinstance(nombres, list):
        return {indice: str(nombre) for indice, nombre in enumerate(nombres)}
    if isinstance(nombres, dict):
        return {int(indice): str(nombre) for indice, nombre in nombres.items()}
    return {}


def resolver_ruta(base: Path, valor: str | None) -> Path | None:
    if not valor:
        return None
    ruta = Path(valor)
    return ruta if ruta.is_absolute() else base / ruta


def revisar_yaml(ruta_yaml: Path) -> tuple[bool, list[str]]:
    mensajes = []
    datos = cargar_yaml(ruta_yaml)
    base = Path(datos.get("path", ruta_yaml.parent))
    if not base.is_absolute():
        base = ruta_yaml.parent / base

    nombres = normalizar_nombres(datos.get("names", {}))
    if not nombres:
        mensajes.append("No se encontraron nombres de clases en 'names'.")
    else:
        mensajes.append(f"Clases detectadas: {', '.join(nombres.values())}")

    split_validacion = datos.get("val", datos.get("valid"))
    splits = {"train": datos.get("train"), "val": split_validacion}
    valido = True
    for nombre_split, valor in splits.items():
        ruta_split = resolver_ruta(base, valor)
        if ruta_split is None or not ruta_split.exists():
            mensajes.append(f"Falta el split '{nombre_split}': {ruta_split}")
            valido = False
            continue
        mensajes.append(f"Split '{nombre_split}' encontrado: {ruta_split}")
    return valido, mensajes


def contar_archivos(carpeta: Path) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for ruta in carpeta.rglob("*"):
        if not ruta.is_file():
            continue
        extension = ruta.suffix.lower() or "[sin_extension]"
        conteo[extension] = conteo.get(extension, 0) + 1
    return dict(sorted(conteo.items(), key=lambda item: (-item[1], item[0])))


def contar_imagenes(carpeta: Path) -> int:
    if not carpeta.exists():
        return 0
    return sum(1 for ruta in carpeta.rglob("*") if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN)


def contar_etiquetas(carpeta: Path) -> int:
    if not carpeta.exists():
        return 0
    return sum(1 for ruta in carpeta.rglob("*.txt") if ruta.is_file())


def encontrar_split_yolo(carpeta: Path, nombres: tuple[str, ...]) -> Path | None:
    for nombre in nombres:
        candidato = carpeta / nombre
        if (candidato / "images").exists() and (candidato / "labels").exists():
            return candidato
    return None


def detectar_estructura_yolo(carpeta: Path) -> tuple[bool, dict[str, Path]]:
    train = encontrar_split_yolo(carpeta, ("train", "training"))
    valid = encontrar_split_yolo(carpeta, ("valid", "val", "validation"))
    test = encontrar_split_yolo(carpeta, ("test", "testing"))
    splits = {}
    if train:
        splits["train"] = train
    if valid:
        splits["valid"] = valid
    if test:
        splits["test"] = test
    return bool(train and valid), splits


def crear_yaml(carpeta: Path, splits: dict[str, Path], salida: Path) -> None:
    nombres = {
        0: "person",
        1: "helmet",
        2: "vest",
        3: "gloves",
        4: "goggles",
        5: "mask",
        6: "safety_shoes",
        7: "no_helmet",
        8: "no_vest",
    }
    datos = {
        "path": str(carpeta.resolve()),
        "train": str((splits["train"] / "images").relative_to(carpeta)),
        "val": str((splits["valid"] / "images").relative_to(carpeta)),
        "names": nombres,
    }
    if "test" in splits:
        datos["test"] = str((splits["test"] / "images").relative_to(carpeta))
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(yaml.safe_dump(datos, sort_keys=False, allow_unicode=True), encoding="utf-8")


def mostrar_conteo(conteo: dict[str, int]) -> None:
    print("Tipos de archivo encontrados:")
    for extension, cantidad in list(conteo.items())[:12]:
        print(f"  {extension}: {cantidad}")


def main() -> None:
    argumentos = leer_argumentos()
    carpeta = Path(argumentos.ruta)
    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")

    print(f"Revisando dataset en: {carpeta.resolve()}")
    conteo = contar_archivos(carpeta)
    mostrar_conteo(conteo)

    archivos_yaml = buscar_yaml(carpeta)
    if archivos_yaml:
        print("\nYAML encontrados:")
        for ruta_yaml in archivos_yaml[:5]:
            print(f"  {ruta_yaml}")
        ruta_yaml = archivos_yaml[0]
        valido, mensajes = revisar_yaml(ruta_yaml)
        print(f"\nYAML recomendado: {ruta_yaml}")
        for mensaje in mensajes:
            print(f"- {mensaje}")
        if valido:
            print("\nEstado: dataset YOLO listo para entrenar.")
            print(f"Comando sugerido: python main.py entrenar --datos {ruta_yaml}")
            return

    estructura_valida, splits = detectar_estructura_yolo(carpeta)
    if splits:
        print("\nEstructura YOLO detectada:")
        for nombre, ruta_split in splits.items():
            imagenes = contar_imagenes(ruta_split / "images")
            etiquetas = contar_etiquetas(ruta_split / "labels")
            print(f"  {nombre}: {imagenes} imagenes, {etiquetas} etiquetas")
    if estructura_valida:
        print("\nEstado: estructura YOLO valida, pero falta YAML.")
        if argumentos.crear_yaml:
            salida = Path(argumentos.salida)
            crear_yaml(carpeta, splits, salida)
            print(f"YAML generado en: {salida}")
            print(f"Comando sugerido: python main.py entrenar --datos {salida}")
        else:
            print("Puedes generar el YAML con:")
            print(f"python main.py revisar-dataset --ruta {carpeta} --crear-yaml")
        return

    print("\nEstado: la carpeta no esta lista para YOLO.")
    print("No encontre data.yaml/epp.yaml ni estructura train/images + train/labels.")
    if conteo.get("[sin_extension]", 0):
        print(
            "Ademas hay muchos archivos sin extension; eso suele indicar una carpeta "
            "equivocada, cache fragmentada o una descarga extraida incorrectamente."
        )
    print("\nEstructura esperada:")
    print("data/raw/nombre_dataset/train/images")
    print("data/raw/nombre_dataset/train/labels")
    print("data/raw/nombre_dataset/valid/images")
    print("data/raw/nombre_dataset/valid/labels")
    print("data/raw/nombre_dataset/data.yaml")


if __name__ == "__main__":
    main()
