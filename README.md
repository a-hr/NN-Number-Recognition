# Project
This repository contains the code necessary to train a convolutional neural network (`model_training.py`) on Python and a toy website to test 
the model yourself (available on the `web/` folder).

You can try out the page by clicking on this [link](https://a-hr.github.io/NN-Number-Recognition/).

## Handwritten number recognition 

On the python file `model_training.py` you can find the code necessary to train a convolutional neural network to be able to 
recognize handwritten numbers in diverse positionings.

## Testing

On the `web/` folder a prototype web page running the trained model is available. You can try and draw on the canvas and let the program 
guess what the number is.

To run the model on a Windows machine, just run `run.bat`. You must have python on your PATH for this script to work. It will automatically 
launch a server with python and redirect your browser to it on your localhost.

## Model export

Once the model is saved as an `.h5` file, tensorflow.js can be used to export our model to web. 
* First, install the package via pip: `pip install tensorflow.js`
* Create a new folder for your model: `mkdir model_folder`
* Finally, run the following code to export your model: `tensorflowjs_converter --input_format keras yourmodel.h5 model_folder`
This will create a number of bin files (depends on your model size) and a `.json` file containing model data. These files can be used
on a web page through JavaScript to run the model, as seen on the toy web page.

## Dependencies
* tensorflow