# ML Engineer Homework Assignment: solution proposed

## Problem

The problem is divided in two tasks:

* Task 1: detection of the 2 most likely clothes categories out of 5 possibles. This is actually a typical multi-class task (there is actually only one real category for a given item picture). For any given item picture, the objective here is to get the probabilities of each of the 5 possible categories (blouses, casual_dresses, mini_dresses, shirts, tank_tops), and select the 2 highest ones. 

* Task 2: detection of 3 potential tags. Here different tags might appear together; this is a typical multi-label task. 

## General presentation of the solution

The strategy selected is a multi-task approach (here 2-tasks): one single model performs both kinds of classification. In this model:

* The task 1 is a multi-class classification of the cc3 classes. The output for this task (see function transfer_learning_model) is a dense(5) layer, to be matched with a one-hot-encoded version of the cc3 columns (containing 5 classes). As it is a multi-class problem, the loss function for that task is a softmax, i.e., the probabilities of all classes sum to 1.

* The task 2 is a multi-label classification of the polka dot, floral, checker columns. The output for this task (see function transfer_learning_model) is a dense(3) layer, to be matched with these 3 columns (already in one-hot-encoded format). As it is a multi-label problem, the loss function for that task is a sigmoid, i.e., the probabilities for each label is independent from the other labels.

The model makes use of a transfer-learning model. It is composed of a convolutional base (VGG16 net), and a dense output for each task. See the model.png file for details.

It is using the tf.keras api, as it has become the standard go-to from TF 2.0.


## Files

Directory structure:
```
.
├── data
└── model_training
    ├── tf_keras_model
    └── saved_model
        └── my_model
            └── 1
                ├── assets
                └── variables
```

Original files (detailed in `homework_dl.md`)

- `data.zip` file containing image files that you will use for prediction (not in git repo as heavy. available here: [https://vinted-ml-homework.s3.eu-central-1.amazonaws.com/OCJ4HAtw0xW/v3.zip](https://vinted-ml-homework.s3.eu-central-1.amazonaws.com/OCJ4HAtw0xW/v3.zip))

- `data.parquet`: a label file

- `test.parquet` : a file for making predictions

- `example_predictions.parquet`: a file with example predictions

Solution files:

- `Vinted_exercise.ipynb` : the main notebook. The notebook is divided in 3 parts:

1. Data loading, exploration and preparation
2. Multi-task Modelling
3. Serving the model using TF-SERVING

- `predictions.parquet`: the file containing the predictions, for the `test.parquet` file.

- `module.py` - a module containing helper functions

- directory `./model_training`: it contains the model run logs. 

- directory `./model_training/tf_keras_model`: it contains the model run with model architecture descriptions:

 - `./model_training/tf_keras_model/model-epoch09-cc3_loss1.2428-tags_loss0.2830-val_cc3_acc0.52-val_tags_acc0.88.h5`: the tf.keras model containing both weights and model architecture. 

 - `./model_training/tf_keras_model/model.png`: the description of the model layers

 - `./model_training/tf_keras_model/model_summary.txt`: complementary description of the model layers, wit total number of parameters in the model

 - `./model_training/tf_keras_model/Loss_Accuracy_plots.png`: plot of the loss and accuracy depending on epochs

- directory `./model_training/saved_model`: the tf.keras model is converted to tensorflow .pb file format. This is used for the tensorflow-serving application.


## Note on software & hardware used

The model was run on an Ubuntu 18.04 machine, using a GTX 1060 GPU. One epoch lasts ~ 90sec. 

The python environment with the version of the packages is detailed in the `conda_list.txt` file. 


## What could be added to this work

- Unit testing

- Use MLflow for consistent model tracking

- Use logging

- Use Tensorboard for performance tracking during a given run

- Hyperparameters tuning using cross-validation

- Implement fine-tuning by unfreezing the last convolutional block of the VGG16

- Try different architectures (tried a few ones, but many more to explore)


