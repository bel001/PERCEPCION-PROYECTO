# Guía Detallada del Código Fuente

Este documento ofrece un desglose técnico de cada módulo del proyecto, explicando sus funciones principales y la lógica de implementación con fragmentos de código.

---

## 1. Punto de Entrada: `main.py`

Este script actúa como la interfaz principal del proyecto. Su objetivo es unificar todas las herramientas bajo un único comando fácil de recordar.

### Gestión Automática de Entorno Virtual
Una característica clave es que el script se asegura de correr siempre en el entorno correcto:

```python
def relanzar_con_venv_si_existe() -> None:
    if not PYTHON_VENV.exists():
        return
    # Si ya estamos en el venv, no hacemos nada
    if Path(sys.prefix).resolve() == (RAIZ_PROYECTO / ".venv").resolve():
        return
    # Si no, relanzamos el script usando el python del venv
    python_venv = PYTHON_VENV
    os.execv(str(python_venv), [str(python_venv), str(Path(__file__).resolve()), *sys.argv[1:]])
```

---

## 2. Configuración y Reglas: `src/configuracion.py`

Define el "cerebro" de las reglas de negocio y estética del sistema.

### Normalización de Etiquetas
El sistema es flexible con los nombres de las clases gracias a esta lógica:

```python
def normalizar_nombre_clase(nombre: str, configuracion: ConfiguracionEPP) -> str:
    # 1. Limpia espacios, convierte a minúsculas y usa guiones bajos
    nombre_normalizado = _normalizar_clave(nombre)
    # 2. Busca en el diccionario de alias (ej: 'hardhat' -> 'helmet')
    return configuracion.alias_clases.get(nombre_normalizado, nombre_normalizado)
```

---

## 3. Preprocesamiento: `src/preprocesar.py`

Prepara datasets crudos para que YOLO pueda aprender de forma óptima.

### Mejora de Imagen con CLAHE
Para que el modelo vea mejor en condiciones de luz difícil, aplicamos esta lógica:

```python
def limpiar_imagen(imagen, tamano_maximo: int, limpiar_ruido: bool):
    # Mejora el contraste usando CLAHE en el canal de luminosidad (LAB)
    lab = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2LAB)
    luminosidad, canal_a, canal_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagen_mejorada = cv2.merge((clahe.apply(luminosidad), canal_a, canal_b))
    return cv2.cvtColor(imagen_mejorada, cv2.COLOR_LAB2BGR)
```

---

## 4. El Motor de Detección: `src/detectar.py`

Es el módulo más complejo. Gestiona la cámara, la inferencia de IA y la lógica de alertas.

### Lógica de Asociación (EPP a Persona)
¿Cómo sabe el programa que un casco pertenece a un trabajador específico? Usa **IoU (Intersection over Union)**:

```python
for elemento in elementos_epp:
    # Busca al trabajador que tenga la mayor superposición con el casco/chaleco
    mejor_estado = max(
        estados,
        key=lambda estado: calcular_iou(elemento.caja, estado.persona.caja),
    )
    # Si la superposición es suficiente, lo asocia a esa persona
    superposicion = calcular_iou(elemento.caja, mejor_estado.persona.caja)
    if superposicion > configuracion.umbrales.iou_asociacion:
        mejor_estado.epp_presente.add(elemento.etiqueta)
```

### Sistema de Alerta Dinámica
Una vez asociados los objetos, se verifica el cumplimiento:

```python
for estado in estados:
    for requerido in configuracion.epp_obligatorio:
        if requerido not in estado.epp_presente:
            # Si falta algo obligatorio, se marca como infracción
            estado.epp_faltante.add(requerido)
```

---

## 5. Entrenamiento y Validación

### `src/entrenar.py`
Utiliza la API de Ultralytics para lanzar el proceso:
```python
modelo = YOLO(argumentos.modelo)
modelo.train(
    data=str(ruta_datos),
    epochs=argumentos.epocas,
    imgsz=argumentos.tamano,
    device=argumentos.dispositivo
)
```

### `src/validar.py`
Calcula el **error aproximado** para dar feedback al desarrollador:
```python
# Un error alto indica que esa clase necesita mas imagenes o mejores etiquetas
error_aproximado = 1.0 - map50_95
```

---

## 📂 Resumen de Flujo de Datos

1.  **Entrada**: `data/raw/sh17/original/` con imagenes y etiquetas.
2.  **Preparacion**: `src/preparar_sh17.py` crea `data/processed/sh17/epp.yaml`.
3.  **Proceso opcional**: `src/preprocesar.py` limpia y normaliza imagenes.
4.  **Entrenamiento**: `src/entrenar.py` genera el modelo `best.pt`.
5.  **Ejecucion**: `src/detectar.py` carga el modelo y aplica las reglas de `reglas_epp.yaml`.
