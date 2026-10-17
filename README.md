# R2-A2-S8S9 · Informe de resultados de tratamiento de imagen

**Percepción Computacional (604011): Especialización en Inteligencia Artificial, Universidad de Cundinamarca**
Autor: **Juan Pablo Cristancho Tula**

Este repositorio analiza cinco fotos del mismo objeto (estuche de carga Hollyland) tomadas bajo condiciones distintas: referencia, baja iluminación, desenfoque, fondo complejo y otra perspectiva. Con esas fotos se diagnostica su calidad con OpenCV y se trata la imagen problemática.

## Resultado principal
Imagen tratada: `oscura.jpg` (subexposición y bajo contraste).
Tratamiento: **gamma adaptativa → CLAHE sobre L (CIELAB) → medias no locales sobre L**.

| Métrica | Antes | Después | Referencia |
|---|---|---|---|
| Brillo | 76.7 | 120.5 | 144.3 |
| Contraste | 25.7 | 33.9 | 37.2 |
| Ruido (Immerkær) | 0.49 | 0.38 | 1.12 |
| Bordes Canny (%) | 0.69 | 3.88 | 3.94 |

## Contenido
- `R2_A2_S8_Tratamiento_de_Imagen_Juan_Pablo_Cristancho.ipynb`: notebook ejecutado con tablas y figuras.
- `tratamiento_imagen.py`: script completo.
- `imagenes/`: las cinco capturas originales sin editar.
- `resultados/`: figuras, imagen tratada, comparación JPG/PNG y `metricas.json`.

## Cómo reproducir
**En Google Colab:** abre el notebook, sube la carpeta `imagenes/` y ejecuta todas las celdas.

**En local:**
```bash
pip install opencv-python numpy matplotlib pandas
python tratamiento_imagen.py --imagenes imagenes --salida resultados
```
