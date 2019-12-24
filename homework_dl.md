# ML Engineer Homework Assignment

## Problem

Vinted members upload images of items that they want to sell. Sellers must enter the category of the item so that we can list it in the correct place in the category tree. Your first objective is to simplify the upload process for the seller by predicting the two most likely categories for uploaded items by using image information. We also want to extract additional information from the image, such as patterns, which are very useful for buyers who search for items. Your second objective is to predict if certain pattern tags (polka dot, floral, and checker) are present in the image. Note that there might not be any pattern tag present or there can be multiple pattern tags present in an image - your model has to support these cases. After you complete your first two objectives, you should have one or two models, depending on your approach – you can use one model to do both tasks (multi-task learning) or train two independent models. Keep in mind that Vinted has a lot of members, so we care a lot about inference speed, model size, and reliability of your solution.

## Data

The dataset is available here: [https://vinted-ml-homework.s3.eu-central-1.amazonaws.com/OCJ4HAtw0xW/v3.zip](https://vinted-ml-homework.s3.eu-central-1.amazonaws.com/OCJ4HAtw0xW/v3.zip)

You are provided with `data.zip` file containing image files that you will use for prediction, a label file - `data.parquet`, a file for making predictions `test.parquet` and a file with example predictions - `example_predictions.parquet`.

- `data.zip` - a zipped folder containing all the images.
- `data.parquet` - a Parquet file containing these columns:
  - `uuid` - unique identifier of the image file. It corresponds to the image file names (without the extension) in the `data.zip` file.
  - `cc3` - the category of the item. Possible values: ['BLOUSES', 'CASUAL_DRESSES', 'MINI_DRESSES', 'SHIRTS', 'TANK_TOPS']
  - `color1` - the color of the item. Possible values: ['APRICOT', 'BLACK', 'BLUE', 'BODY', 'BROWN', 'BURGUNDY', 'CORAL', 'CREAM', 'DARK-GREEN', 'GOLD', 'GREEN', 'GREY', 'KHAKI', 'LIGHT-BLUE', 'LILAC', 'MINT', 'MUSTARD', 'NAVY', 'ORANGE', 'PINK', 'PURPLE', 'RED', 'ROSE', 'SILVER', 'TURQUOISE', 'VARIOUS', 'WHITE', 'YELLOW']
  - `polka dot` - the presence of a polka dot pattern tag. Possible values: [0, 1]
  - `floral` - the presence of a floral pattern tag. Possible values: [0, 1]
  - `checker` - the presence of a checker pattern tag. Possible values: [0, 1]
- `test.parquet` - a Parquet file containing these columns:
  - `uuid` - the unique identifier of the image file. It corresponds to the image file names (without the extension) in the `data.zip` file.
- `example_predictions.parquet` - a Parquet file containing these columns:
  - `uuid` - the unique identifier of the image file. This column must be the same as in the `test.parquet`.
  - `category1` - the prediction for the most likely category for this item.
  - `category2` - the prediction for the second most likely category for this item.
  - `polka dot` - the prediction of the presence of a polka dot pattern tag. Possible values: [0, 1]
  - `floral` - the prediction of the presence of a floral pattern tag. Possible values: [0, 1]
  - `checker` - the prediction of the presence of a checker pattern tag. Possible values: [0, 1]

## Requirements

- Do an exploratory data analysis of the provided dataset.
- Build a data preprocessing pipeline, which can be used for your machine learning models.
- Build a machine learning model that classifies the images into one of five different categories.
- Build a machine learning model that detects pattern tags in the images. You can use the same model as for the point above if you do multi-task learning.
- Use the above models to predict the two most likely categories and the presence of the pattern tags for the items in the `test.parquet` file. Export these predictions to `predictions.parquet` file.
- You should have a single Jupyter notebook, which shows all of your work. You can also optionally put helper functions, classes, etc. in separate Python modules and import them in this notebook. Submit this notebook with the output cells present.
- You should have weight files for each of your trained models (or one file if you have just one model). There should be a clear and easy way to load these files to your Jupyter notebook and use them for predictions. You can optionally save the model architecture together with the weights.
- Use Python 3 in this project. This requirement doesn't apply to libraries written in a different language.
- Commit your solution to a private Github repository (they are free) and invite `mokahaiku` Github user as a collaborator.

## Suggestions

- Your solution should use Python software development best practices.
- There should be an easy way to run your solution.
- README is the perfect place to provide instructions on how to run your solution.
- The solution should be local, i.e. it should not make network requests when doing inference (making predicitons).
- Short documentation of design decisions and assumptions can be provided in the notebook itself or in the README file.
- Time is not constrained, feel free to solve it whenever you're able to. Just don't forget to communicate with us if you can't find a free evening within a couple of weeks :-)

## Evaluation Criteria

The solution is evaluated based on these criteria (in order):

1. The demonstration of your deep learning skills.
1. The demonstration of your machine learning skills.
1. The demonstration of your data analysis skills.
1. Your knowledge of software engineering best practices.
1. The top 1 accuracy and top 2 accuracy on the category prediction measured on `test.parquet`.
1. The precision, recall and F1 score on the pattern tag prediction measured on `test.parquet`.
1. The combined inference speed of your models - if you have a single model the inference time counts only once.

## Bonus Points

We understand that you have skills that you couldn't show us doing this homework, so we provide some ideas for what you can do to show off your skills. These bonus points are not mandatory - do them only if they capture your strengths or if you have extra time to spare on this task and want to try something new. These bonus tasks are mutually independent and not in order. Here are the bonus tasks:

- Prepare the model/models for inference by creating a web service (preferably RESTful) with the model/models. You can use any tool that you see fit for this task such as Django, Flask, TensorFlow Serving, etc.
