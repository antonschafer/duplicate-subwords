# On the Effect of (Near) Duplicate Subwords in Language Modelling

Code to the paper [On the Effect of (Near) Duplicate Subwords in Language Modelling](https://arxiv.org/abs/2404.06508). The code is based on the [Languini Kitchen](https://github.com/languini-kitchen/languini-kitchen), a codebase for training language models. For an overview of the changes made to support vocabulary (de)duplication, see [this diff](https://github.com/antonschafer/duplicate-subwords/pull/1/files) or check out the [(de)duplication implementation](./languini/de_duplication/mappings.py).

## Reproducing Plots
To reproduce the plots from the paper without retraining models, you can load the relevant results via
```
wget https://y5d6.c15.e2-3.dev/public-bucket/results.zip
unzip results.zip
```
then install languini in a new environment
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -e . --upgrade
```
and run the [analysis notebook](analysis.ipynb).

## Training Models

You can also train models under the (de)duplicated settings of the paper:

1. Setup the environment as described above and follow the [instructions for obtaining the dataset](#download-and-tokenise-the-books3-dataset).
2. Train models as described in Languini. Configure duplication settings via the config arguments
    - `frac_duplicated`: fraction of the vocabulary to duplicate
    - `p_duplicate`: probability of a token being a duplicated token (given that the corresponding vocabulary item is duplicated)
    - `dedup_type`: type of deduplication to apply to vocabulary ("whitespace", "lower", "plural", "all") or ""/None for no deduplication. Use the suffix "_50%" to only deduplicate half of the respective near duplicates.
    - `embed_noncanonical`: whether to add an extra embedding indicating whether a token is "non-canonical"
    
    For example, to train a model that coresponds to the $p(c) + e_\text{non-canonical}$ with  $\mathbb{S}_\text{all}$ entry in Table 5, run
    ```
    TRAIN_STEPS=18265
    ACC_STEPS=8 # this works for a 4090 (24 GB)

    torchrun --standalone languini/projects/gpt/main.py small \
        --train_batch_size 128 \
        --gradient_accumulation_steps $ACC_STEPS \
        --decay_steps $TRAIN_STEPS \
        --max_train_steps $TRAIN_STEPS \
        --frac_duplicated 0  \
        --dedup_type "all" \
        --embed_noncanonical \
        --seed 0
    ```
3. Evaluate the model as described in Languini. You can additionally specify which deduplication mapping $\mathbb{S}$ to use for the projected perplexity $\mathrm{PPL}_\mathbb{S}$ via the `eval_dedup_type` argument. E.g., to evaluate the run above, run
    ```
    RUN_PATH="path/of/your/wandb/run" # alternatively specify checkpoint_file and config_file

    ./venv/bin/torchrun --standalone languini/projects/gpt/eval.py \
        --wandb_run $RUN_PATH \
        --eval_data_split test \
        --eval_dedup_type all \
        --last_n 128
    ```
    if you specify a wandb run, this will automatically load the checkpoint from the run and finally upload the results to the run's summary.

## Other Experiments
You can also reproduce the GLUE experiments via e.g.
```
./finetune_glue.sh path/of/your/wandb/run 
```

or the Word2Vec experiments via e.g.
```
python -m languini.projects.word2vec.main --frac_duplicated 1.0 --p_dup 0.5
```


