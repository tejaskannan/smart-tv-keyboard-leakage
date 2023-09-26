# Suggestions Model
This module uses a Random Forest classifier to predict when a keyboard uses suggestions given a move count sequence. This code describes how to train a new instance of this suggestions model.

We include the final trained model in the file `suggestions_model.pkl.gz` for reproducibility reasons. Note that the remainder of the code will use the path `suggestions_model/suggestions_model.pkl.gz` to fetch the suggestions model. If you want to reference new model parameters, you should place it at this path.

## Training
We train the suggestions model on a set of *simulated* move count sequences. We simulate suggestions using the most common subsequence characters in an English Dictionary fit from the Wikipedia words. The training script is `determine_autocomplete.py`. An example for training a new model is below.
```
python determine_autocomplete.py --english-words-file <PATH-TO-GDRIVE>/word-lists/wikipedia-words.txt --passwords-file <PATH-TO-GDRIVE>/word-lists/rockyou.txt --dictionary-file <PATH-TO-GDRIVE>/dictionaries/english/wikipedia.db --output-file model.pkl.gz --word-count 5000
```
The variable `<PATH-TO-GDRIVE>` refers to the location where you downloaded the ![Google Drive folder](https://drive.google.com/drive/folders/1iBWbk8wqRq2OYdgXhRM71CzBnK5pXcJ3?usp=sharing) containing word lists and dictionaries. Model training should take no more than a minute. The output of this command is the model weights saved in the `--output-file` (e.g., `model.pkl.gz`) above.

## Evaluation
The script `evaluate.py` tests the trained model on a set of move count sequences from English words taken from the Samsung Smart TV. In this setting, we are no longer simulating suggestions--we use the suggestions as given by the target system. You must provide the trained model parameters (the pickle file from the previous step) to the evaluation script. The command below shows an example.
```
python evaluate.py --model-path model.pkl.gz
```
The accuracy on this set should be over `90%`.
