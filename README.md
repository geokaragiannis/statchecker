# IEA factchecking project

## Template and row_index predictions

* `src/tokenizer/tokenizer_driver.py`
    - tokenizes input data, using `spacy`
* `src/featurizer/featurizer_extractor.py`
    * Can either be `tfidf` or `word-embeddings`
    * Each object will contain the features as an instance variable
* `src/featurizer/sentence_embedding.py`
    * Implements a `scikit-learn` transformer, which gives us the word (`glove`) embeddings, using `spacy`. I use average pooling here.
* `src/classifier/classifier_linear_svm.py`
    * Contains the Linear `SVM` model along with the sigmoid on top of it, which gives us calibrated probabilities for the `topn` predictions
    * The instance variable `cv`, determines the number of the cross validation folds. Hence, our input dataset needs to have **at least `cv` number of samples per class**, otherwise it will **not** work.
    * The default `cv` is 3, but one could try bigger values (4, 5), and see which gives better results experimentally.
* `src/parser/dataset_parser.py`
    * Contains logic, which creates the dataset for `template` and `row_index` predictions.
    * Look at `src/templates/template_transformer.py` for how the templates are created from the original Excel formulas.
    * Also has logic for combining features of sentences and claims
    * This class is a little "messy", which could be fixed a little later
* `src/templates/template_transformer.py`
    * Logic for creating templates from Excel formulas
    * Uses various Regexes (look at `src/regex/regex.py`), to filter (to some extent) Excel formulas
    * Uses `pandas` `apply` function to create a series of transformations for each row of the `DataFrame`

## Running Experiments

The experiments supported right now are `src/experiments/exp_only_row_idx.py` and `src/experiments/exp_only_templates.py`. You can see the code in both, to better see how I use the above classes for `Tokenization`, `Feturization` and `Classification`. The code is not very clean, and I expect to make it better as I go on.

### How to run:

You need to have:  

(1) the data sepcified on top of each experiment python file. I.e the variable `DATA_PATH`, needs to correspond to a file. 
(2) The `csv` file needs to have the same format as the currect on this repo (`data/main_annotated_dataset_12-16-2019.csv`).
(3) Need to have all the requirements from `requirements.txt`

Before you can run do:

* Create a Virtual Environment with python version `3.5.6` (although probably anything above that will do)
* `python -m spacy download en_core_web_md`
* `pip install -r requirements.txt`

From the root path of the directory run:

`python -m src.experiments.exp_only_templates --num_runs 1 --cv 3 --min_samples_per_label 20 --topn 3`

Where:

* `num_runs`: The number of times the task is run. The end accuracy numbers are the average of the accuracy of each run
* `cv`: Cross validation folds used for `Classification`
* `min_samples_per_label`: Min number of samples we keep for each label. **Note that `min_samples_per_label` >= `cv`**
* `topn`: Topn predictions to return
