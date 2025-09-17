# Iterative Feedback-based Time-Series Anomaly Detection with Adaptive Diffusion Models

This is an repository hosting the code of our paper:  Iterative Feedback-based Time-Series Anomaly Detection with Adaptive Diffusion Models



## Datasets

1. PSM (PooledServer Metrics) is collected internally from multiple application server nodes at eBay.
   You can learn about it
   from [Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://dl.acm.org/doi/abs/10.1145/3447548.3467174)
   .

## Usage

### Environment

Install Python 3.8.

```python
pip install -r requirements.txt
```

By default, datasets are placed under the "tf_dataset" folder. If you need to change 
the dataset, you can modify the dataset path  in the json file in the "config" folder. 
Here is an example of modifying the training dataset path:

```json
"datasets": {
    "train|test": {
        "dataroot": "tf_dataset/smap/psm_train.csv",
    }
},
```

### Training
Next, we demonstrate using the psm dataset.

#### We use dataset psm for training demonstration.

```python
# Use time_train.py to train the task.
# Edit json files to adjust dataset path, network structure and hyperparameters.
python time_train.py -c config/psm_time_train.json
```

### Test
The trained model is placed in "experiments/*/checkpoint/" by default. 
If you need to modify this path, you can refer to "config/psm_time_test.json":

```json
"path": {
  "resume_state": "experiments/psm_TRAIN_128_2048_100/checkpoint/E100"
},
```
 
#### We also use dataset psm for testing demonstration.

```python
# Edit json to adjust pretrain model path and dataset_path.
python time_test.py -c config/psm_time_test.json
```

