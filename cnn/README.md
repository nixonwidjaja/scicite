# Convolutional Neural Network for SciCite Text Classification

- `cnn.ipynb`: Evaluation of the CNN model, using a combination `keras` layers such as `Embeddings`, `BatchNormalization`, `Dropout`, `Conv1D`, `Dense`, and `MaxPooling1D`. 
- `cnn_tests.ipynb`: Consists of the trained CNN model's performance tested against different text variations.
- `scicite_cnn.keras`: The trained CNN model with 250 epochs, 256 batch size, and 0.0005 learning rate.

To run the `cnn.ipynb` properly, do the following:

1. Install the python requirements using ```pip install -r requirements.txt```
2. Download the GLoVe Embeddings from this [link](!http://nlp.stanford.edu/data/glove.6B.zip). The current model utilizes `glove.6B.100d.txt`
3. You're set! Training of the model can be done locally. With the current settings, it takes about 100 minutes.