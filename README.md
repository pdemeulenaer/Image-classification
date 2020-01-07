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

Directory structure: **(new since 2020-01-07: model 2 available)**
```
.
├── data
└── model_training
    ├── tf_keras_model
    ├── tf_keras_model_1
    ├── tf_keras_model_2
    └── saved_model
        └── my_model
            ├── 1
            │   ├── assets
            │   └── variables
            └── 2
                ├── assets
                └── variables

```

Original files (detailed in `homework_dl.md`)

- `data.zip` file containing image files that you will use for prediction (not in git repo as heavy. available here: [https://vinted-ml-homework.s3.eu-central-1.amazonaws.com/OCJ4HAtw0xW/v3.zip](https://vinted-ml-homework.s3.eu-central-1.amazonaws.com/OCJ4HAtw0xW/v3.zip))

- `data.parquet`: a label file

- `test.parquet` : a file for making predictions

- `example_predictions.parquet`: a file with example predictions

Solution files:

- `Vinted_exercise.ipynb` : the main notebook. 

- `predictions.parquet`: the file containing the **predictions of model 1**, for the `test.parquet` file.

- `predictions_model2.parquet`: **the file containing the predictions of model 2 (new since 2020-01-07: model 2 available)**, for the `test.parquet` file.

- `module.py` - a module containing helper functions

- directory `./model_training`: it contains the model run logs. 

- directory `./model_training/tf_keras_model_1`: it contains the model run with model architecture descriptions:

 - `./model_training/tf_keras_model_1/model-epoch09-cc3_loss1.2428-tags_loss0.2830-val_cc3_acc0.52-val_tags_acc0.88.h5`: the tf.keras model containing both weights and model architecture. 

 - `./model_training/tf_keras_model_1/model.png`: the description of the model layers

 - `./model_training/tf_keras_model_1/model_summary.txt`: complementary description of the model layers, wit total number of parameters in the model

 - `./model_training/tf_keras_model_1/Loss_Accuracy_plots.png`: plot of the loss and accuracy depending on epochs

- directory `./model_training/saved_model`: the tf.keras model is converted to tensorflow .pb file format. This is used for the tensorflow-serving application.

**new since 2020-01-07: model 2 available:**

- directory `./model_training/tf_keras_model_2/` containing model 2:

 - `./model_training/tf_keras_model_2/model-epoch32-cc3_loss1.1084-tags_loss0.3010-val_cc3_acc0.59-val_tags_acc0.89.h5`: the tf.keras model containing both weights and model architecture. 


## Note on the general run of the notebook: (**new since 2020-01-07**)

The notebook is divided in 3 parts:

1. Data loading, exploration and preparation
2. Multi-task Modelling
3. Serving the model using TF-SERVING

In part 1, the datasets are 
- loaded, 
- prepared (label encoding and categorical encoding), uuid column completed with ".jpg" format
- analysed: simple EDA shows data imbalance. Correlation matrix shows the potential of using a multi-task model since some correlations exists between cc3 and tags features (example: shirt-checker correlation)
- downsampled (if that option is turned on, like in model 1. In model 2 it is turned off)

In part 2, the multi-task model is 
- declared (with all model hyperparameters, as well as options like data augmentation)
- trained. 
Note that 4 datasets generators are created:
- the train dataset
- the validation dataset
- the heldout dataset: this is a dataset from the same dataframe as train and validation, that is held out in order to build performance metrics
- the test dataset, on which the model is applied. The results will be in predictions_model2.parquet (for model 1) or predictions_model2.parquet (for model 2)
Note that the results of the model (plots, checkpoints, ...) will be saved in the (empty) tf_keras_model directory. When a model is satisfactory, please rename the folder to tf_keras_model_N (N being index of your model) and re-create a tf_keras_model folder for potential next run.

For the 3rd part, the serving part, it should be activated by activating first the cells of part 1 beforehand. No need to activate cells of part 2 (no need to train a new model, just load the latest one saved).

Part 3 saves the keras model in the saved_model folder. Then (after some inspection of the model), the model can be served using TF-serving. The principle is to launch a TF-serving server that loads the model and listens to requests. Do this by the bash command (open terminal in main directory for this):

tensorflow_model_server --model_base_path=$(pwd)/model_training/saved_model/my_model/ --rest_api_port=9000 --model_name=my_model

Note that the model that will be loaded will be **the latest version**, hence here model 2


## Note on the differences between models 1 and 2 folder (**new since 2020-01-07**)

- Model 1 has a VGG16 convolutional base, while model 2 has VGG19

- While model 1 has input image height x width: 301x217, model 2 has squared input image 224 x 224 (to follow standard practice)

- Model 1 starts with a default learning of 1e-3 for the Adam optimizer, and has been ran for 15 epochs. Model 2 starts with a lower learning rate of 1e-4 and has been ran for 40 epochs. 

- Model 1 uses data downsampling: by downsampling the "blouse" class of the cc3 feature so that all classes have same sample number. Using performance metrics, it can be noticed that the model 1 is performing too poorly for that class (and much better on the other ones). As shown by confusion matrix analysis, the model 2 performs more evenly on all classes. That is the main argument in favour of the second model. (Note: both float precision and downsampling options are configurable in the cell "Main parameters of project" in part 1.)


## Note on performance metrics (**new since 2020-01-07**)

- Added confusion matrix for cc3 classes

- Added ROC and Precision-Recall curves for tags binary labels, as well as confusion matrices for each tag. Probability threshold adapted to 0.4, as it was experienced to be more adapted than 0.5 (more balanced). 


## Note on the precision type adopted (**new since 2020-01-07**)

- Inplemented mixed precision training, i.e. the training is using Float16 precision in place of float32 when possible (especially meaningful during convolution operations). If trained on RTX graphic cards (possessing Tensor Cores) the model could potentially be accelerated twice. In any case, the model is twice lighter compared to case when it has been ran using float32 precision. Note that the input and output layers of the network should be kept in dtype=tf.float32 in TF2.0. From TF2.1 that won't be a requirement anymore. 


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
