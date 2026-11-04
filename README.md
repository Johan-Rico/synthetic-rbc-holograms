# Generación de hologramas de eritrocitos a partir de mapas de fase mediante una GAN

Proyecto final de Deep Learning. Se adapta una GAN condicional para generar el holograma de un eritrocito a partir de su mapa de fase, y se evalúa la consistencia de los hologramas generados reconstruyendo su fase con la transformada de Fourier.

## Objetivos

**General.**

Adaptar y evaluar una red generativa adversaria (GAN) condicional para generar hologramas de eritrocitos a partir de mapas de fase, evaluando su consistencia mediante reconstrucción de fase basada en la transformada de Fourier.

**Específicos.**
1. Preparar los pares de mapas de fase y hologramas del dataset, verificando su correspondencia y normalización y definiendo los conjuntos de entrenamiento, validación y prueba.
2. Implementar y entrenar una GAN condicional, compuesta por un generador y un discriminador, para aprender la correspondencia entre los mapas de fase de entrada y los hologramas asociados.
3. Evaluar los hologramas generados comparándolos con los de referencia y reconstruyendo sus mapas de fase con la FFT, cuantificando el error respecto a las fases de entrada.

## Contenido

```
synthetic-rbc-holograms.ipynb   notebook principal
splits/                         partición train/val/test por holograma original
requirements.txt                dependencias (PyTorch se instala aparte)
.gitignore                      excluye datos, resultados y checkpoints
```

## Datos

El dataset **no se incluye en el repositorio**.

- Fuente: Castañeda, R., Trujillo, C., Doblas, A. (2024). *A human erythrocytes hologram dataset for learning-based model training*. Data in Brief 54, 110424. https://doi.org/10.1016/j.dib.2024.110424
- Descarga: Open Science Framework, DOI 10.17605/OSF.IO/8P7BA
- Licencia: CC BY-NC 4.0 (uso no comercial, con atribución).

Colocar las carpetas así (o ajustar `RUTA_DATASET` en el notebook):

```
data/Holograms/<subcarpetas>/*.png
data/Phase/<subcarpetas>/*.png
```

## Instalación

1. Instalar PyTorch con el comando de https://pytorch.org según el sistema y la GPU.
2. `pip install -r requirements.txt`
3. Abrir el notebook y ajustar `RUTA_DATASET` y `RUTA_SALIDA` en la sección 0.

Comprobar que PyTorch ve la GPU:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```