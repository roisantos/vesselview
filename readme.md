

Para configurar una loss function, cambia dos cosas:
el config.json
la opción en la llamada a run_benchmark.py en el script .sh:
srun python ./code/training/run_benchmark.py -model "RoiNet_Dice"

Para lanzar FRNet  (la red original, con bloques RRCNNBlock y RecurrentConvNeXtBlock), uso el script fr.sh
Para lanzar FRNetR (la red original, con bloques ResidualBlock), uso el script frr.sh
Para lanzar RoiNet  uso los scripts roiscript.sh ,roiscriptx15.sh  (multiplica x 1.5),roiscriptx20.sh  (multiplica x 2.0). 
(RoiNetR funciona pero no la uses, ni siquiera la pensé, le correspondería el script rr.sh)  (Nota: Los nombres de los modelos (con R o sin R) no tienen lógica, lo ves más claro en el config.json)

En cualquier caso, estoy usando kernels de 9x9 o 11x11. Probé con 5x5 y 3x3 y no subia de DICE 0.72 ... también es cierto que tenía otras cosas de la configuración, debería volver a probar. Esto es lo que pasa cuando no tienes tiempo para hacer una búsqueda de hiperparámetros . Pierdes más tiempo, no eres metódico... a ver cómo hacemos el ablation study.

Con una sóla máquina (32 cores. 64 GB de RAM y una GPU) me entrena RoiNet x 1.5 (18 Millones de parámetros) . Pero si subo a x 2.0 (será más de 25 Millones de parámetros) ya me da error out of memory (OOM).


1. Arquitectura y Configuración
train_multiGPU.py: Principal script de entrenamiento que configura y ejecuta la red en múltiples GPUs si están disponibles. Incluye la lógica para gestionar los dispositivos, cargar datasets, inicializar el modelo (FRNet) y aplicar técnicas de optimización y programación de tasa de aprendizaje.
settings_benchmark.py: Define las configuraciones de modelos y facilita la creación de instancias de FRNet.
script_slurm.sh: Script SLURM para ejecutar el código en el entorno FinisTerrae, especificando los recursos (4 GPUs, 48 núcleos de CPU, 96GB de RAM).

2. Datasets y Preprocesamiento
dataset.py y resizeDatasetTo512.py: Scripts para gestionar y preparar los datos. resizeDatasetTo512.py redimensiona las imágenes a 512x512, un tamaño manejable para el entrenamiento inicial.
SegmentationDataset: Clase personalizada para cargar imágenes y etiquetas, aplicando transformaciones, padding y normalización. Se asegura de que las dimensiones sean múltiplos de 32 para optimizar el procesamiento en el modelo.

3. Modelos y Funciones de Pérdida
FRNet: Implementación de la red convolucional para segmentación de alta precisión.
DiceLoss (en loss.py): Función de pérdida basada en el coeficiente de Dice, ideal para tareas de segmentación, ya que mide la superposición entre las predicciones y las etiquetas reales.

4. Entrenamiento y Evaluación
run_benchmark.py: Ejecuta el entrenamiento y evalúa el rendimiento de FRNet en los conjuntos de datos definidos. Almacena los resultados de las métricas en formato JSON y guarda el modelo con el mejor índice de Dice.
custom_collate y traverseDataset: Funciones auxiliares para manejar lotes de datos y realizar el entrenamiento por épocas. traverseDataset incluye lógica de evaluación y cálculo de métricas para validar el modelo en cada época.

5. Visualización y Monitoreo de Activaciones y Gradientes
utils.py: Contiene funciones para registrar activaciones y gradientes en cada capa convolucional durante el entrenamiento. Las activaciones y gradientes se guardan en un SummaryWriter de TensorBoard, permitiendo el análisis visual de las capas y posibles cuellos de botella.
plot_activations.ipynb: Un notebook para visualizar las activaciones y gradientes, permitiendo identificar patrones y evaluar la eficacia del modelo en diferentes épocas.

6. Inferencia
infer.py y infer_linux.py: Scripts de inferencia que cargan el modelo entrenado y procesan un directorio de imágenes para generar predicciones de segmentación. Aplican padding a las imágenes para asegurar que las dimensiones sean compatibles con el modelo y guardan los resultados como tensores y archivos de imagen.

7. Evaluación de Métricas
evaluation.py: Calcula métricas de rendimiento como precisión (accuracy), sensibilidad (sens), especificidad (spe), y coeficiente de Dice. Estas métricas permiten analizar el rendimiento del modelo en términos de aciertos y errores en la segmentación de vasos.

8. Documentación
readme.md: Explica la estructura y el propósito del código, junto con las instrucciones de ejecución. Describe el proceso para añadir nuevos datasets y modelos, lo que facilita la extensión del proyecto.

Enjoy!!