# WADM
wide-attention and deep model

## Installation
- Python 2.7   
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19

## Train the model

**Run command below to train the model:**
- Training baseline single DL model based on CPeMS dataset.
```
python train_t.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.

- Training composite DL model based on CPeMS dataset.
```
python train_wd.py --model model_name
```
You can choose "w_attention_d" (WADM) or "wd_crossLayer_attention" (DCN) as arguments. The ```.h5``` weight file was saved at model folder.

- Training model based on FBBC dataset.
```
python train_bike.py --model model_name
```
You can choose "lstm", "gru" as arguments for training single DL model or choose "w_attention_d" (WADM) for training composite DL model.

## Experiment
Data are obtained from the Caltrans Performance Measurement System (CPeMS) and Fremont Bridge Bicycle Counter (FBBC).
```
device: GTX 1050
dataset: CPeMS and FBBC
optimizer: RMSprop
```
