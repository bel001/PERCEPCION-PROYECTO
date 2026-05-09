# Carpeta de datasets

El dataset no esta incluido en el proyecto.

Usa estas carpetas asi:

- `raw/`: pega aqui los datasets descargados manualmente.
- `processed/`: aqui se genera el dataset limpio despues de ejecutar `src.preprocesar`.
- `samples/`: puedes guardar imagenes pequenas para pruebas rapidas.

Flujo:

```bash
python -m src.preprocesar --entrada data/raw --salida data/processed --sobrescribir
```

Pesos aproximados de las fuentes usadas en la documentacion:

- Hugging Face PPE Detection: 668 MB.
- Mendeley PPE 5-Class: 124 MB.
- Roboflow Hard Hat Universe: peso no publicado; contiene 7,036 imagenes.
- Mendeley Dataset of PPE: 236 MB.
- CHV / Real-time PPE dataset: 440 MB.

Reserva al menos 1.47 GB para los comprimidos conocidos, mas espacio adicional
para descomprimir y generar `data/processed/`.
