To build the suggestions model, we run:
 build_model_sim_new(ms_dict_rockyou, englishDictionary, ss_path : str, words,
                    include_rockyou : bool = False, bin_transform : List[int] = [], weight : int = 3,
                    mistakes : bool = False, max_depth : int = 3)

This function builds a random forest model.  For features, the model uses histograms of move
sequences with bins for moves of different lengths.


The arguments are:
ms_dict_rockyou - A premade dictionary mapping strings from the rockyou password list
to their move sequences on the non-suggestions Samsung Smart TV keyboard.

englishDictionary - A dictionary of English words that is used for predicting suggestions
in order to simulate move sequences on a suggestions keyboard. 

ss_path - The path to a json file that hardcodes the Samsung Smart TV suggestions for
single charactar strings.

words - A list of English words on which to train the model.

include_rockyou - A boolean flag indicating whether or not to include the Rockyou
strings in training.

bin_transform - A list of 10 integers that indicates how to map move lengths to bins.
The integer at any index i in bin_transform denotes which bin a move of length i should
got to (and any move of length >= 9 goes to the bin in index 9).

weight - A number indicating which strategy to use when transforming a move sequence
to a histogram.  More detail on this is below.

mistakes - A boolean flag indicating whether or not to add simulated mistakes to the
training set.  If set, we add either 0, 1, 2, or 3 extraneous moves to the move
sequences with decreasing probability. 

max_depth - The max_depth of the trees in the random forest.



A note on weight; the simplest method of transforming move sequences into histograms
would be to loop through the move sequence and increment the appropriate bin at each
iteration.  This is what is performed when we use weight=0.  However, we theorize
that moves later in the sequence are more suggestive of whether suggestions are being
used.  This is because suggestions should generally increase the number of moves that
are length 0 or 1, and Samsung's suggestion system should be more accurate when it has
more data, so we expect a larger proportion of those moves later in the string.  Thus, 
we tested strategize that don't simply increment the bins, but increase them by a number
related to the placement of a given move in the move sequence.  Through testing, we 
concluded that this did improve our model, and we use weight=3, which gives each move
a weight proportional to its index in the move sequence.




Once a model is created using this function, we save to a .pkl.gz file using
save_model().