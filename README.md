## Overview

This code repository provides the data and evaluation script to help you start your assignment in Relation Extraction.

## Data

Both train and (unlabeled) test set are provided. It's stored in `data/` folder. The data format and semantic meaning of all relations are specified in `data/SemEval2010_task8_training/`. 

## Evaluation

The offical evaluation script provided in `data/SemEval2010_task8_scorer-v1.2/`, which gives the detailed report for multiple evaluation metrics. Note that there are three ways of evaluation in official script, which is documented in `data/SemEval2010_task8_scorer-v1.2/README.txt`. For our assignment, the performance will be graded on the third method: macro-avergaed (9+1)-way classification, with direction taken into account.

To run the offical evaluation script, you need to install `perl`, which won't be too hard if following the guideline [here](https://www.perl.org/get.html). After installation, you can test the performance using:

```bash
perl semeval2010_task8_scorer-v1.2.pl <PROPOSED_ANSWERS> <ANSWER_KEY>
```

Besides the offical script, we also provide a non-standard implementation `python`. To run the evaluation result, directly run the following script to get an idea of what the result is like:

```bash
python utils/scorer.py
```

, or you can call function `evaluate(y_true, y_pred)` directly in other python program.
