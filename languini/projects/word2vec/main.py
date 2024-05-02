import os
import itertools
import argparse

import pickle
import gensim
from tqdm import tqdm
import torch

from languini.dataset_lib import languini_books
from languini.common_lib import experiment_utils
from languini.de_duplication.mappings import configure_dedup_mapping
from languini.train_lib import train_utils


def main():
    # arguments, see languini/projects/gpt/configs.py for meaning of each argument
    parser = argparse.ArgumentParser()
    # duplication args
    parser.add_argument('--frac_duplicated', type=float, default=0.0)
    parser.add_argument('--p_dup', type=float, default=0.0)
    parser.add_argument('--duplication_seed', type=int, default=0)
    # word2vec args
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--window', type=int, default=5) # corresponds to default in gensim
    parser.add_argument('--sample', type=float, default=0.001) # corresponds to default in gensim
    parser.add_argument('--alpha', type=float, default=0.025) # corresponds to default in gensim
    parser.add_argument('--min_alpha', type=float, default=0.0001) # corresponds to default in gensim
    parser.add_argument('--negative', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    # data args -- do not change
    parser.add_argument('--train_steps', type=int, default=18265)
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=512)
    # output args
    parser.add_argument('--output_dir', type=str, default=None)
    
    # setup config
    config = parser.parse_args()
    config.dataset = f"books_{config.vocab_size}"

    # setup output dir
    if config.output_dir is None:
        config.output_dir = f"word2vec_results/{config.frac_duplicated}_{config.p_dup}_{config.duplication_seed}"
    os.makedirs(config.output_dir, exist_ok=False)

    vocab_mapping = configure_dedup_mapping(
        sp=train_utils.load_tokeniser(config),
        frac_duplicated=config.frac_duplicated,
        p_duplicate=config.p_dup,
        dedup_type="",
    )
    
    # load data
    END_OF_DOC_TOKEN = 2
    train_ds = languini_books.LanguiniDatasetIterator(
        data_path=os.path.join("data/books", config.dataset),
        split='train',
        repeat=True,
        global_batch_size=config.batch_size,
        batch_idxs=list(range(config.batch_size)),
        micro_batches=1,
        sequence_length=config.sequence_length,
        device="cpu",
        end_of_doc_token=END_OF_DOC_TOKEN,
        vocab_mapping=vocab_mapping,
    )
    train_ds = itertools.islice(train_ds, config.train_steps)
    
    # convert to list of "sentences" for gensim
    # treat each sequence as a sentence and each token as a word
    sentences = []
    print("Processing dataset ...")
    for batch_x, batch_y, is_padded, _ in tqdm(train_ds, total=config.train_steps):
        assert not is_padded
        new_sentences = batch_x.squeeze(0).tolist()
        sentences += new_sentences
    
    # train word2vec
    w2v_model = gensim.models.Word2Vec(
        min_count=1,
        window=config.window,
        vector_size=config.dim,
        sample=config.sample,
        alpha=config.alpha,
        min_alpha=config.min_alpha,
        negative=config.negative,
        workers=config.workers,
        seed=config.seed,
    )
    print("Building vocab ...")
    w2v_model.build_vocab(tqdm(sentences))
    print("Training word2vec model ...")
    w2v_model.train(tqdm(sentences), total_examples=len(sentences), epochs=1)
    
    print("Saving ...")
    # gather embeddings
    input_embeddings = torch.full((vocab_mapping.output_vocab_size, config.dim), fill_value=float("nan"))
    output_embeddings = torch.full((vocab_mapping.output_vocab_size, config.dim), fill_value=float("nan"))
    for subword_id in w2v_model.wv.index_to_key:
        input_embeddings[subword_id] = torch.tensor(w2v_model.wv[subword_id])
        output_embeddings[subword_id] = torch.tensor(w2v_model.syn1neg[w2v_model.wv.key_to_index[subword_id]])
    
    # save
    torch.save(input_embeddings, os.path.join(config.output_dir, "input_embeddings.pt"))
    torch.save(output_embeddings, os.path.join(config.output_dir, "output_embeddings.pt"))
    w2v_model.save(os.path.join(config.output_dir, "word2vec.model"))

    with open(os.path.join(config.output_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print("Done.")

if __name__ == "__main__":
    main()