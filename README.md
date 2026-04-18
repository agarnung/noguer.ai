# noguer.ai

Joder, una IA que dibuja como Miguel Noguera.

![cristo_skater](assets/cristo_skater.png)

**Nota**: este proyecto nace de la voluntad de capturar la esencia visual y conceptual del universo de Miguel Noguera mediante un modelo basado en difusión, para dar vida a mis propias ideas (sin dibujar yo necesariamente, vaya).

Ver los pesos del modelo entrenado en [Hugging Face](https://huggingface.co/agarnung/noguera-lora). La [demo en Hugging Face Spaces](https://huggingface.co/spaces/agarnung/noguer.ai) aún no está funcional por falta de memoria GPU libre.

## Introducción técnica

Para entender cómo funciona [FLUX.1 (y familiares)](https://arxiv.org/pdf/2506.15742), el modelo que usaremos para generar imágenes que "recuerden" a las ilustraciones de Miguel Noguera, debemos conocer el concepto de **Flow Matching**. A diferencia de los modelos de difusión tradicionales (como [Stable Diffusion](https://arxiv.org/pdf/2112.10752)), que predicen el ruido que hay que eliminar, FLUX.1 utiliza una formulación de probabilidad basada en ODEs (Ecuaciones Diferenciales Ordinarias).

La técnica, que es el *flow matching* con *timestep sigmoid*, permite que el modelo aprenda la **trayectoria más eficiente** desde una distribución de ruido Gaussiano hacia la distribución de datos (i.e. las imágenes del _dataset_ de Noguera). Matemáticamente, esto se traduce en una reducción de la varianza durante el entrenamiento, permitiendo que un rango alto de [LoRA](https://arxiv.org/pdf/2106.09685) (256 dimensiones) (técnica de PEFT o fine-tuning eficiente) capture detalles hiper-específicos del trazo y la composición sin colapsar el espacio latente (i.e. sobresimplificación o aprendizaje nulo o baldío).

## Algunos ejemplos:

<table border="1" style="border-collapse: collapse; text-align: center; width: 100%;">
  <tr>
    <td>
      <img src="./assets/samples/1.png" alt="1.png" style="width: 250px;" /><br>
      <i>Cristo boxeando con Lucifer</i>
    </td>
    <td>
      <img src="./assets/samples/2.png" alt="2.png" style="width: 250px;" /><br>
      <i>Un tío haciendo un kickflip con una rampa en una rampa</i>
    </td>
    <td>
      <img src="./assets/samples/3.png" alt="3.png" style="width: 250px;" /><br>
      <i>Una señora rescatando a un gato de un árbol</i>
    </td>
    <td>
      <img src="./assets/samples/4.png" alt="4.png" style="width: 250px;" /><br>
      <i>Un teléfono prehistórico hecho de madera</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./assets/samples/5.png" alt="5.png" style="width: 250px;" /><br>
      <i>-</i>
    </td>
    <td>
      <img src="./assets/samples/6.png" alt="6.png" style="width: 250px;" /><br>
      <i>Un zapato jubilado recordando sus mejores pisadas</i>
    </td>
    <td>
      <img src="./assets/samples/7.png" alt="7.png" style="width: 250px;" /><br>
      <i>-</i>
    </td>
    <td>
      <img src="./assets/samples/8.png" alt="8.png" style="width: 250px;" /><br>
      <i>En el infierno también venden helados</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./assets/samples/9.png" alt="9.png" style="width: 250px;" /><br>
      <i>-</i>
    </td>
    <td>
      <img src="./assets/samples/10.png" alt="10.png" style="width: 250px;" /><br>
      <i>Cristo Mal</i>
    </td>
    <td>
      <img src="./assets/samples/11.png" alt="11.png" style="width: 250px;" /><br>
      <i>Cristo Mal</i>
    </td>
    <td>
      <img src="./assets/samples/12.png" alt="12.png" style="width: 250px;" /><br>
      <i>-</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./assets/samples/13.png" alt="13.png" style="width: 250px;" /><br>
      <i>Cuando gritaba sonaba como una campana metálica muy estridente</i>
    </td>
    <td>
      <img src="./assets/samples/14.png" alt="14.png" style="width: 250px;" /><br>
      <i>Cuando gritaba sonaba como una campana metálica muy estridente</i>
    </td>
    <td>
      <img src="./assets/samples/15.png" alt="15.png" style="width: 250px;" /><br>
      <i>Cristo haciendo un unboxing de su propia resurrección</i>
    </td>
    <td>
      <img src="./assets/samples/16.png" alt="16.png" style="width: 250px;" /><br>
      <i>Cristo haciendo un unboxing de su propia resurrección</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./assets/samples/17.png" alt="17.png" style="width: 250px;" /><br>
      <i>La plaza de un pueblo en un concierto de jazz asistido por cuatro viejos con muletas</i>
    </td>
    <td>
      <img src="./assets/samples/18.png" alt="18.png" style="width: 250px;" /><br>
      <i>La plaza de un pueblo en un concierto de jazz asistido por cuatro viejos con muletas</i>
    </td>
    <td>
      <img src="./assets/samples/19.png" alt="19.png" style="width: 250px;" /><br>
      <i>Un rascacielos levantado en un nenúfar</i>
    </td>
    <td>
      <img src="./assets/samples/20.png" alt="20.png" style="width: 250px;" /><br>
      <i>Un elefante montando a caballo</i>
    </td>
  </tr>
</table>

## Configuración del entrenamiento

El modelo actual es el resultado de un entrenamiento intensivo (usando el framework [Ai-Toolkit](https://github.com/ostris/ai-toolkit) y una [NVIDIA Quadro RTX A6000 48 GB](https://www.nvidia.com/es-es/products/workstations/rtx-a6000/)) con los siguientes parámetros:

- **Arquitectura**: FLUX.1-dev cuantizado (T5 para el text encoder).
- **Adaptación**: LoRA con Rank 256 y Alpha 256. La alta dimensión del rango permite capturar la "lógica absurda" y personalidad característica de las composiciones.
- **Optimizador**: AdamW 8-bit con un *learning rate* de $5e-05$.
- **Dataset**: 586 imágenes procesadas con *buckets* multiresolución (hasta 1024px).
- **Flujo de difusión**: *flowmatch* con *timestep sigmoid* a lo largo de 10.000 pasos.

## Implementación en ComfyUI

Para la inferencia utilizamos [ComfyUI](https://www.comfy.org/), una de las herramientas modernas más potentes que que existen hoy día, que para muchos es desconocida porque está muy ligada al mundo artístico pero que, como backend y configurador de flujos de trabajo con muchísimos modelos de IA de todo tipo, es muy potente (no cabe más que decir que permite la creación de nodos con E/S propios totalmente a libre albedrío en Python).

Para replicar el estilo del modelo entrenado con total fidelidad, el flujo de trabajo en ComfyUI (ver ejemplo en la siguiente imagen) debe estar sincronizado con la configuración del entrenamiento.

![comfy](assets/comfy.png)

Para lograr esta sincronización, se siguieron varios pasos:

### 0. Creación del dataset

Todos los pares de texto-imagen se colocan en el volumen del backend de AI-Toolkit (e.g. `app/storage/datasets`) para no tener que subirlos una a una (en mi caso, extraje el texto de las imágenes mediante OCR con un modelo multimodal):

<div align="center">
  <img src="assets/cap1.png" alt="cap1" width="800px" />
</div>

Y la salida de los pesos del modelo entrenado, junto con su configuración y los _checkpoints_ se guardarían automáticamente en `output/`. El árbol de directorios se vería algo así:

<div align="center">
  <img src="assets/cap4.png" alt="cap4" width="300px" />
  <img src="assets/cap2.png" alt="cap2" width="300px" />
  <img src="assets/cap3.png" alt="cap3" width="300px" />
</div>

En pleno entrenamiento, se pueden configurar muestras de validación cada ciertas iteraciones, para monitorizar cómo va evolucionando:

<div align="center">
  <img src="assets/cap5.png" alt="cap5" width="800px" />
</div>

### 1. Sincronización del sampler y scheduler

**Configuración obligatoria**:
- **Nodo Sampler**: usar `SamplerCustomAdvanced` con el sampler `euler`.
- **Nodo Scheduler**: configurar `BasicScheduler` en modo `simple` o `beta`.
- **Nodo ModelSamplingFlux**: configurar `max_shift: 1.15` y `base_shift: 0.5`. Este nodo es crítico para ajustar el comportamiento del modelo a resoluciones de 1024x1024.

### 2. Gestión del LoRA y pesos

**Recomendaciones de uso**:
- **Strength Model**: debido al Rank 256 y los 10k pasos, el modelo está muy saturado. Se recomienda un valor de **0.7 a 0.8** para evitar artefactos.
- **Strength CLIP**: mantener en **1.0**. El entrenamiento no tocó el *text encoder*, por lo que dependemos del CLIP original para entender el prompt.

### 3. Parámetros de flujo

- **Guidance (FluxGuidance)**: está bien fijarlo exactamente en **4.0**, valor por defecto del entrenamiento. FLUX maneja el *guidance* de forma distinta a SDXL; valores superiores a 7.0 pueden romper la imagen.
- **Negative Prompt**: no se debería utilizar acondicionamiento negativo, pues FLUX no lo requiere. Por el contrario, habría que usar un `BasicGuider` conectado directamente al prompt positivo.

## Opciones de despliegue e inferencia

Valoré varias alternativas. Ojalá tenerlo en una aplicación corriendo en CPU pero estos modelos de difusión se van de madre. Aun así, las alternativas reales son:

**A) Inferencia en local (con GPU)**: 
E.g. usando una instancia de ComfyUI propia para tener control total sobre los nodos de *shift* y demás. Se puede encontrar un ejemplo del flujo aquí: [flux-dev-basic.json](assets/flux-dev-basic.json). Hay mucho hueco para optimización del modelo, pero no tengo mucho tiempo para iterar, así que los resultados son los que son.

**B) Inferencia en PCs sin GPU (CPU-Only)**: 
Para usuarios con hardware limitado, las opciones son:
- **Quantización GGUF/EXL2**: reducir los pesos a 4-bit o 8-bit.
- **OpenVINO**: convertir el modelo a ONNX para aceleración específica en procesadores Intel.
- **Modelos destilados (LCM)**: aplicar una destilación para generar imágenes en solo 4 u 8 pasos.

**C) Inferencia _serverless_**:
Para mostrar "al público":
- Crear una instancia en la nube accesible por web, mediante pago de GPU por uso (e.g. Droplets de Digital Ocean).
- Uso de servicios dedicados a este tipo de POCs como los **Spaces de Hugging Face** (lo que se hace aquí), Civitai o Glif. Aunque en [esta demo](https://huggingface.co/spaces/agarnung/noguer.ai) se utiliza una versión del modelo (FLUX.1-schnell) más rápida y menos precisa para que queda en las memoria de acceso libre de los espacios de Hugging Face.

> [!INFO]
> La GPU dinámica (o **ZeroGPU**) es un sistema de Hugging Face que te presta una tarjeta gráfica potente de forma gratuita solo durante los segundos que tarda en ejecutarse el script de _demoo_. Al terminar el proceso, la GPU se libera para otro usuario, permitiéndote ofrecer un modelo pesado como FLUX sin tener que pagar una instancia dedicada 24/7.

## Bonus: recomendaciones de estilo

Para forzar a **FLUX** a ignorar el fotorrealismo y adoptar un estilo de **ilustración pura**, mediante ensayo y error han sido efectivas las siguientes recomendaciones. El objetivo es conseguir un **estilo intrínseco**, de manera que el modelo convierta al trazo del lápiz en la **norma visual** cada vez que se use el _trigger_ (si hay), capturando la presión del grafito y la textura del carbón sin necesidad de comandos o _prompts_ adicionales:

**Eliminación del realismo**:
* **Caption Dropout Rate**: bajar a **0.0** obliga al modelo a leer las etiquetas el 100 % de las veces.
* **Guidance Scale**: reducir de **4.0** a **2.0 - 3.0**; valores altos limpian la imagen y eliminan la textura orgánica del lápiz.
* **Regularización**: desactivarla, pues no queremos que el modelo intente recordar cómo es una foto real.
* **Use EMA**: **activado**; es importante para estabilizar texturas granulares o finas, como las del grafito.

**Aumentar la memoria del estilo**:
* **Linear Rank**: subir de **32** a **128 o 256**. Un rango alto es necesario para codificar detalles complejos como los rasgos de los personajes o el trazo.
* **Steps**: aumenta a **5000 o 6000**; este estilo de dibujo requiere más tiempo para "fijarse" como el estándar intrínseco del modelo.
* **Learning Rate**: mantener en **0.0001** (o subir a **0.0002** si el cambio no es drástico); la diea es dejar que el estilo se "cocine" con más pasos en lugar de mucha potencia de golpe.

## Referencias

1. [**Lipman, Y., et al. (2023)**: *Flow Matching for Generative Modeling*](https://arxiv.org/pdf/2210.02747).
2. [**Esser, P., et al. (2024)**: *Scaling Rectified Flow Transformers (FLUX.1 technical report)*](https://arxiv.org/pdf/2403.03206).
3. [**Vila, J. L. (2015)**: *Entre el humor y la filosofía: algunas ideas sobre las ideas de Miguel Noguera*](https://dialnet.unirioja.es/servlet/articulo?codigo=5923236).
4. Ver [este trabajo](https://www.domenecmirallestagliabue.com/miguel_noguera) del artista Domenec Miralles.

## TODO

- El modelo entrenado es una primera iteración; es muy simple, estaría bien recopilar muchos más dibujos, incluso a color, de los vídeos de los Ultrahsows, etc.
