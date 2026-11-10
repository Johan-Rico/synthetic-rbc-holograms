# Generación de hologramas de eritrocitos a partir de mapas de fase mediante una GAN

Proyecto final de Deep Learning. Se entrena una GAN condicional (pix2pix) que recibe el mapa de fase de un eritrocito y genera su holograma. Para medir la calidad, se reconstruye la fase del holograma generado con la transformada de Fourier (FFT) y se compara con la fase de entrada.

## Objetivos

**General.** Adaptar y evaluar una GAN condicional para generar hologramas de eritrocitos a partir de mapas de fase, evaluando su consistencia mediante reconstrucción de fase basada en la FFT.

**Específicos.**
1. Preparar los pares de mapas de fase y hologramas, verificando su correspondencia y normalización y definiendo los conjuntos de entrenamiento, validación y prueba.
2. Implementar y entrenar una GAN condicional (generador y discriminador) que aprenda la correspondencia entre fase y holograma.
3. Evaluar los hologramas generados comparándolos con los de referencia y reconstruyendo sus fases con la FFT, cuantificando el error respecto a las fases de entrada.

## Contenido

```
synthetic-rbc-holograms.ipynb   notebook principal, se ejecuta de arriba hacia abajo
splits/                         partición train/val/test por holograma original
requirements.txt                dependencias (PyTorch se instala aparte)
.gitignore                      excluye datos, resultados y checkpoints
```

El notebook crea `resultados/<nombre_del_experimento>/` con:

```
config.json      configuración usada
historial.csv    pérdidas y métricas de cada época
muestras/        imágenes de fases, generados y reales
checkpoints/     generadores guardados y ultimo.pt para retomar
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

Detalles del uso de los datos:
- Imágenes de 256 × 256 píxeles, normalizadas a [-1, 1].
- La partición es 70 % / 15 % / 15 % (train / val / test) **por holograma original**, para que recortes del mismo holograma no queden en conjuntos distintos.
- Se usa solo la versión original de cada recorte, sin las versiones rotadas o volteadas del dataset, para no repetir imágenes.
- Los hologramas tienen franjas en una de dos diagonales. Con `diagonal="voltear"`, los de la otra diagonal se voltean junto con su fase, de modo que todos queden con la misma.

## Instalación

1. Instalar PyTorch con el comando de https://pytorch.org según el sistema y la GPU.
2. `pip install -r requirements.txt`
3. Abrir el notebook y ajustar `RUTA_DATASET` y `RUTA_SALIDA` en la sección 0.

Comprobar que PyTorch ve la GPU:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Modelo

- **Generador:** U-Net de 8 niveles con conexiones de salto. Entra la fase y sale el holograma.
- **Discriminador:** PatchGAN de 70 × 70. Recibe fase y holograma concatenados y decide, por parche, si el par es real o generado.
- **Pérdida del generador:**

$$\mathcal{L}_G = \mathcal{L}_{GAN} + \lambda_{L1}\,\mathcal{L}_{L1} + \lambda_{fase}\,\mathcal{L}_{fase}$$

  con $\lambda_{L1}$ = `lambda_l1` y $\lambda_{fase}$ = `lambda_fase`.

## Hiperparámetros

Todos están en `CFG` y se guardan en `config.json` de cada experimento.

| Parámetro | Valor | Qué es |
|---|---|---|
| `nombre` | `pix2pix_diag_fase` | Nombre del experimento y de su carpeta de salida. Cambiarlo en cada experimento nuevo, porque si no se sobrescribe. |
| `semilla` | 0 | Semilla aleatoria, para poder repetir el entrenamiento. |
| `batch` | 8 | Pares por lote. |
| `ngf` | 64 | Canales base del generador. Más canales dan un modelo más grande. |
| `ndf` | 64 | Canales base del discriminador. |
| `lr_G` | 2e-4 | Tasa de aprendizaje del generador. |
| `lr_D` | 2e-4 | Tasa de aprendizaje del discriminador. |
| `beta1` | 0.5 | Parámetro beta1 del optimizador Adam. |
| `lambda_l1` | 100 | Peso de la L1 frente a la pérdida adversaria. Más alto, el generador se parece más al holograma real píxel a píxel. |
| `lambda_fase` | 6 | Peso de la pérdida de fase. Con 0 se desactiva y queda el pix2pix original. |
| `diagonal` | `voltear` | `voltear` usa todos los pares y voltea los de la otra diagonal. `filtrar` usa solo los que ya están en una diagonal. |
| `n_val_epoca` | 256 | Pares de validación con los que se miden las métricas al final de cada época. |
| `guardar_cada` | 5 | Cada cuántas épocas se guarda el generador. |

## Métricas

Los hologramas están en [-1, 1]. Las métricas con "fase" reconstruyen la fase del holograma con la FFT: se filtra el pico lateral del espectro, se lleva al centro y se toma el ángulo.

### Pérdidas durante el entrenamiento

| Métrica | Significado |
|---|---|
| `loss_D` | Pérdida del discriminador. Si baja a casi 0, el discriminador le gana al generador. |
| `loss_G_gan` | Pérdida adversaria del generador: qué tanto logra engañar al discriminador. |
| `loss_L1` | Error absoluto medio entre el holograma generado $\hat{h}$ y el real $h$: $\mathcal{L}_{L1}=\frac{1}{N}\sum_p \lvert \hat{h}_p-h_p\rvert$. Menor es mejor. |
| `loss_fase` | Pérdida de consistencia de fase: $\mathcal{L}_{fase}=1-\left\lvert \frac{1}{N}\sum_p e^{\,i(\hat{\varphi}_p-\varphi_p)}\right\rvert$, con $\hat{\varphi}$ la fase recuperada del holograma generado y $\varphi$ la fase de entrada. Va de 0 (fase igual) a 1 (sin relación). Ignora un desplazamiento global de las franjas. |
| `D_real`, `D_falsa` | Probabilidad media que el discriminador asigna a pares reales y generados. Valores cercanos a 0.5 indican un entrenamiento equilibrado. |

Cada pérdida de validación (`val_L1`, `val_L_fase`) es la misma cantidad calculada sobre validación. Comparada con la de entrenamiento muestra el sobreajuste.

### Evaluación en validación y prueba

| Métrica | Significado |
|---|---|
| `mae` | Error absoluto medio entre generado y real: $\frac{1}{N}\sum_p \lvert \hat{h}_p-h_p\rvert$. Es la L1. |
| `psnr` | Relación señal a ruido pico, en dB. Mayor es mejor: $\mathrm{PSNR}=10\log_{10}\!\left(\frac{4}{\mathrm{MSE}}\right)$, porque el rango de las imágenes es 2. |
| `prom_real`, `prom_gen` | Prominencia del pico lateral del espectro $F$: $\dfrac{\max \lvert F\rvert}{\mathrm{mediana}\,\lvert F\rvert}$, es decir, su altura sobre el nivel de fondo (sin contar la zona del pico central). Mide qué tan marcadas están las franjas. El generado debe parecerse al real. |
| `dist_real`, `dist_gen` | Distancia del pico lateral al centro del espectro, en píxeles. Mide la frecuencia de las franjas. |
| `r2_real`, `r2_gen` | R² entre la fase reconstruida y la de entrada. 1 es un ajuste perfecto. Se calcula con el ajuste de abajo, así que no penaliza escala, desfase constante ni inclinación. |
| `nrmse_real`, `nrmse_gen` | Raíz del error cuadrático medio de ese ajuste, normalizada por la amplitud de la fase de entrada. 0 es perfecto, y menor es mejor. |

Para `r2` y `nrmse` se ajusta la fase reconstruida $\varphi_{rec}$ a la fase de entrada $\varphi_{ref}$ por mínimos cuadrados:

$$\varphi_{rec} \approx a\,\varphi_{ref} + b + c\,x + d\,y$$

Sea $r$ el residuo del ajuste y $\tilde{\varphi}_{rec}=\varphi_{rec}-(b+c\,x+d\,y)$ la fase reconstruida sin desfase ni inclinación. Entonces:

$$R^2 = 1-\frac{\mathrm{Var}(r)}{\mathrm{Var}(\tilde{\varphi}_{rec})}, \qquad \mathrm{NRMSE}=\frac{\sqrt{\overline{r^2}}}{\lvert a\rvert\,\sigma(\varphi_{ref})}$$

Los valores de `r2_real` y `nrmse_real`, calculados con el holograma real, son el **piso**: lo mejor que puede dar la reconstrucción con FFT. El generador se lee frente a ese piso. Como **control**, se reconstruye cada holograma contra la fase de otro recorte, y el resultado debe ser claramente peor que el piso.

## Referencias

- Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix). Implementación de referencia: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-cyclegan-and-pix2pix).
- Castañeda, R., Trujillo, C., Doblas, A. (2024). Dataset de hologramas de eritrocitos, citado arriba.
