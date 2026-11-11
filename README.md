# Fashion Classification

A deep learning project based on the fashion classification lessons in [DataTalks.Club's Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). It uses TensorFlow and Keras to adapt an ImageNet-pretrained Xception network to classify clothing photographs into 10 categories.

The work lives in [fashion-classification.ipynb](fashion-classification.ipynb), which walks through transfer learning, hyperparameter experiments, checkpointing, and training-curve analysis.

## Dataset

Images are organized by split and class under `datasets/`:

| Split | Images | Purpose |
| --- | ---: | --- |
| Training | 3,068 | Fit the classification head |
| Validation | 341 | Compare experiments and select checkpoints |
| Test | 372 | Reserved for final evaluation |

The class names, in the directory order used by Keras, are:

```text
dress, hat, longsleeve, outwear, pants,
shirt, shoes, shorts, skirt, t-shirt
```

`outwear` preserves the spelling used in the dataset. These are color clothing photographs, rather than the grayscale Fashion-MNIST dataset.

```text
fashion-classification/
|-- datasets/
|   |-- train/<class>/*.jpg
|   |-- validation/<class>/*.jpg
|   `-- test/<class>/*.jpg
|-- fashion-classification.ipynb   # Training and experiments
|-- main.py                        # Placeholder entry point
|-- pyproject.toml                 # Python requirements and dependencies
|-- uv.lock                        # Locked dependency versions
|-- model_v1.weights.h5            # Weights-only snapshot
`-- xception_v1_*.h5               # Full-model checkpoints
```

## Getting started

Use Python 3.12 (the version selected in `.python-version`) and [uv](https://docs.astral.sh/uv/). Run the following commands from the project root:

```bash
uv sync --locked
uv run jupyter notebook fashion-classification.ipynb
```

The environment includes TensorFlow, NumPy, pandas, Matplotlib, and Jupyter. The dataset folders are tracked in this repository; confirm they are present before running the notebook. Internet access is needed on the first run to download Xception's pretrained weights and ImageNet labels.

Select the project's Python environment as the notebook kernel and execute cells from top to bottom. Run Jupyter from the project root so that the relative dataset paths resolve correctly. The notebook redefines `make_model` as the experiments progress, so cell order matters.

Running all cells trains several models, including 30- and 50-epoch experiments, and can take substantial time on a CPU. The saved notebook output reports that GPU acceleration was unavailable in its native Windows environment. `main.py` currently prints a greeting; use the notebook for training.

## Training workflow

1. **Explore an image and pretrained predictions.** Resize a sample to 299 x 299 and inspect the original ImageNet classifier's predictions.
2. **Build a clothing classifier.** Use Xception with `include_top=False`, freeze the backbone, and add global average pooling followed by a 10-unit output layer. Training images are resized to 150 x 150 with batches of 32.
3. **Compare learning rates.** Train with Adam at `0.001` and `0.01` and plot training and validation accuracy.
4. **Save checkpoints.** Monitor validation accuracy and save a full model whenever it improves. A separate cell saves a weights-only snapshot.
5. **Explore the classification head.** Compare dense layers with 10 and 100 ReLU units, then dropout rates of `0.0` and `0.2`.
6. **Run a longer experiment.** Train a head with 100 hidden units and dropout `0.2` for 50 epochs at learning rate `0.001`.

All clothing models use Xception's `preprocess_input` and categorical cross-entropy with `from_logits=True`. Their outputs are logits; apply softmax when probabilities are needed.

The section titled "Data Augmentation" currently has its flip option commented out, so it applies preprocessing without random augmentation. The Xception backbone remains frozen throughout the implemented experiments.

## Recorded results

The existing notebook output and checkpoint filenames record these validation results:

| Experiment | Validation accuracy |
| --- | ---: |
| Best checkpoint in the checkpointing run (`xception_v1_08_0.830.h5`) | About 83.0% |
| Best epoch in the final 50-epoch run | 84.16% |
| Final epoch of the 50-epoch run | 81.23% |

These are results from saved runs, not guarantees for a new execution. No random seed is fixed. The final run does not use the checkpoint callback, so its best epoch is not automatically saved. The notebook does not yet evaluate the held-out test set.

## Predict a clothing category

With `xception_v1_08_0.830.h5` available in the project root, run this in a notebook cell. Replace `image_path` with a clothing image:

```python
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

model = keras.models.load_model(
    "xception_v1_08_0.830.h5", compile=False
)
classes = sorted(p.name for p in Path("datasets/train").iterdir() if p.is_dir())

image_path = "datasets/train/shoes/2dfbea36-6802-473c-b948-60a1e6181534.jpg"
image = keras.utils.load_img(image_path, target_size=(150, 150))
batch = np.expand_dims(keras.utils.img_to_array(image), axis=0)
batch = preprocess_input(batch)

logits = model.predict(batch, verbose=0)
probabilities = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))[0]
index = int(np.argmax(probabilities))
print(f"{classes[index]}: {probabilities[index]:.1%}")
```

Keep the class order consistent with `train_ds.class_indices` when using a different dataset. `decode_predictions` is used only for the original ImageNet model; the clothing classifier uses the 10 labels above.

A full `xception_v1_*.h5` checkpoint can be loaded directly as shown. `model_v1.weights.h5` contains only weights and requires recreating the matching architecture before calling `load_weights`.

## Acknowledgments

This project follows the deep learning material in [Machine Learning Zoomcamp by DataTalks.Club](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/08-deep-learning). Credit goes to the course authors and the original clothing dataset contributors. See the course materials for the dataset's source and usage terms.
