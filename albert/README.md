# ALBERT Model for Scicite Text Classification

- `albert.ipynb`: Evaluation of multiple ALBERT models
- `pipeline.py`: Python script for training, evaluating, and saving the ALBERT model
- `train_script.sh`: Shell script for training in SOC compute cluster
- `utils.py`: Contains various utility functions used in training

To run the `albert.ipynb` properly, do the following:

1. Install the python requirement using ```pip install -r requirements.txt```
2. Download our trained models (albert1.pth, albert2.pth, albert3.pth, albert4.pth, and albert5.pth) in https://www.kaggle.com/datasets/farreldwireswara/scicite-text-classification-with-albert