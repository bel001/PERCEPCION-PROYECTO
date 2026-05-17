# Detalles Técnicos y Arquitectura del Código

Este documento describe las tecnologías utilizadas y la lógica interna detrás del sistema de detección de Equipo de Protección Personal (EPP).

## 🛠️ Tecnologías Utilizadas

El proyecto se basa en un stack moderno de visión artificial y procesamiento de datos:

*   **Lenguaje:** [Python 3.10+](https://www.python.org/)
*   **Modelo de Detección:** [Ultralytics YOLOv11](https://docs.ultralytics.com/) - Utilizado para la detección de objetos en tiempo real (personas y equipo).
*   **Procesamiento de Imágenes:** [OpenCV](https://opencv.org/) - Gestión de frames de cámara, dibujo de cajas de texto y preprocesamiento.
*   **Backend de IA:** [PyTorch](https://pytorch.org/) - Motor de ejecución para las redes neuronales.
*   **Gestión de Datos:** [NumPy](https://numpy.org/) y [YAML](https://yaml.org/) para configuración y manejo de matrices de píxeles.
*   **Interfaz CLI:** `argparse` para la gestión de comandos desde la terminal.

---

## 🏗️ Estructura del Código

El código está organizado de forma modular para separar las responsabilidades del pipeline:

### 1. Punto de Entrada (`main.py`)
Centraliza todos los comandos. Actúa como un orquestador que invoca a los módulos internos dependiendo de la acción solicitada (`preprocesar`, `entrenar`, `validar`, `camara`, `detectar`).

### 2. Capa de Configuración (`src/configuracion.py`)
Utiliza `dataclasses` y `YAML` para gestionar:
*   **Umbrales:** Confianza del modelo e IoU (Intersection over Union).
*   **Reglas de Negocio:** Define qué EPP es obligatorio y qué alias tienen las clases (ej: `hardhat` -> `helmet`).
*   **Estética:** Colores de las cajas y escalas de texto para la visualización.

### 3. Procesamiento y Limpieza (`src/preprocesar.py`)
Implementa técnicas de mejora de imagen antes del entrenamiento:
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Para mejorar la visibilidad en condiciones de luz difícil.
*   **Filtros de Ruido:** Reducción de ruido gausiano y digital.
*   **Normalización:** Asegura que todas las etiquetas YOLO sigan el mismo formato y clases.

### 4. Lógica de Detección y Asociación (`src/detectar.py`)
Es el núcleo dinámico del sistema. Su flujo es:
1.  **Inferencia:** El modelo YOLO detecta todas las clases (personas, cascos, chalecos, etc.).
2.  **Asociación Espacial:** Para cada persona detectada, el sistema busca objetos de EPP dentro de su caja delimitadora o en un radio cercano (usando IoU y centros de masa).
3.  **Verificación de Reglas:** Contrasta el equipo encontrado contra la lista `epp_obligatorio` definida en `reglas_epp.yaml`.
4.  **Sistema de Alerta:** Si falta algún elemento, el estado del trabajador cambia a "Infracción" y se activa la alerta visual.

---

## 🧠 Lógica de Verificación (EPP)

A diferencia de un detector simple que solo cuenta objetos, este sistema aplica lógica asociativa:

```mermaid
graph TD
    A[Captura de Frame] --> B[Inferencia YOLOv11]
    B --> C{¿Hay Personas?}
    C -- No --> D[Mostrar Frame Limpio]
    C -- Sí --> E[Para cada Persona...]
    E --> F[Buscar EPP asociado espacialmente]
    F --> G{¿Cumple reglas_epp.yaml?}
    G -- Sí --> H[Estado: SEGURO (Verde)]
    G -- No --> I[Estado: PELIGRO (Rojo)]
    H --> J[Dibujar Interfaz y Alertas]
    I --> J
```

## ⚙️ Configuración Dinámica

El sistema permite cambiar el comportamiento sin tocar el código Python editando `configs/reglas_epp.yaml`:

*   **`epp_obligatorio`**: Lista de identificadores que el sistema exigirá.
*   **`alias_clases`**: Mapeo para que el sistema entienda diferentes datasets (ej: si un dataset llama al casco `safety-helmet`, el sistema lo mapea a `helmet`).
*   **`clases_negativas`**: Permite detectar explícitamente la ausencia (ej: detectar `no-helmet` para una alerta más rápida).

---

Para mas detalles sobre como usar cada comando, consulta el [Manual de Usuario](documentacion/README.md).
