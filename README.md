# Sketch-Generation-Tool

This project builds upon the Sketch-RNN 'A Neural Representation of Sketch Drawings' (Ha, David and Eck, Douglas 2017) to create
an application which can be used to reinterpret or complete sketches based on a user's preferences. Specifically, a user will draw
a sketch and have it reinterpreted or completed by a pre-trained model. The user will then be presented with a grid of sketches
generated based on their drawing. By choosing their favourite, the user then has another grid presented to them based on their
favourite. The user may continue this process until a satisfactory sketch is found.

The repository also contains a file which, given a dataset and a pre-trained model, apply tranformations to the dataset and carry
out transfer learning using this new dataset.

Project written for Python 3.9.

## To run:
### Sketching Application:
1. If specific pre-trained model desired, change constant *MODEL_DIR* to the url containing the model checkpoint. Default model trained on owl QuickDraw! dataset (https://quickdraw.withgoogle.com/data).
2. Run *SketchTool.py*
3. Sketch and generate!

### Transfer Learning Module:
1. If specific pre-trained model to be used, change constant *MODEL_DIR* to the url containing the model checkpoint. Default model trained on Aaron Koblin Sheep dataset (https://github.com/hardmaru/sketch-rnn-datasets/tree/master/aaron_sheep).
2. Run *TransferLearning.py*. Use flag 'new' to train a model from scratch, otherwise transfer learning used to train using provided model. Use flag 'scale' to apply a horizontal scaling transformation to the dataset before the rotation is applied (default is rotation only).
