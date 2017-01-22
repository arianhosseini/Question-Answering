# Question Answering and Reading Comprehension
=========================================

This package contains code for the *Exploration of various deep neural networks for Question Answering.*

## Running and Implementation details

With different configs for different models in `config` directory, you will be able to train a model on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

To train a model you should run `python train.py name_of_config` (no **.py** extension for config name). Before that you should get the SQuAD data and tokenize it using the `tokenize_data` function in `utils.py` file. This file also contains functions to create a vocabulary, compute length coverege, etc.

All the models are in the `model` directory, from the basic seq2seq model with attention to the [Match-LSTM](https://arxiv.org/abs/1608.07905) model with pointer networks.

[Fuel](http://fuel.readthedocs.io/en/latest/) is being used for the data pipeline. `Iterator` and `Dataset` classes for SQuAD and CNN/DailyMotion datasets are available in `data.py` as well as a Toy Dataset for debugging purposes.

*Evaluation extensions* for models in `lmu_extensions.py` are implemented to be used with blocks' *extensions*.
For seq2seq models average recall, average precision, macro F1, average F1 and exact match accuracies are reported.




## Some cool heatmaps of the Match-LSTM model

The heatmap shows the **attention paid** to each token of the **question** at each step of encoding the **paragraph**.
![Heatmap1](https://s20.postimg.org/9mdbm0hul/heatmap1.png)


![Heatmap2](https://s20.postimg.org/489y0q2ql/heatmap2.png)


## Accumulated Answer Length Coverge

As you can see the SQuAD dataset contains some answers with high lengths which will evidently make sequence to sequence solutions challenging for this task.
![Answer length](https://s20.postimg.org/du83urja5/answerlengths.png)
(hotizontal axis is the length and vertical axis is the acc coverage)


## Acknowledgments
We would like to thank the developers of Theano, Blocks and Fuel at MILA for their excellent work.
