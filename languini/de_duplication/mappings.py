import random
from abc import ABC, abstractmethod

import torch
import numpy as np

from languini.train_lib.train_utils import load_tokeniser


class VocabMapping(ABC):
    """
    Base class for duplication and deduplication mappings
    """
    @abstractmethod
    def __call__(self, x):
        """
        Applies the mapping to the input(s). Can be stochastic (e.g. for duplication)
        """
        pass
    
    @abstractmethod
    def deduplicate_logits(self, logits):
        """
        Deduplicates logits, used for evaluation
        """
        pass

    @abstractmethod
    def deduplicate_labels(self, labels):
        """
        Deduplicates labels, used for evaluation
        """
        pass

    @property
    @abstractmethod
    def input_vocab_size(self):
        """
        Returns the vocabulary size the mapping operates on
        """
        pass

    @property
    @abstractmethod
    def output_vocab_size(self):
        """
        Returns the vocabulary size the mapping maps to
        """
        pass

    @property
    @abstractmethod
    def dedup_vocab_size(self):
        """
        Returns the vocabulary size after deduplication
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Unique identifier for the mapping and its parameters
        """
        pass

    def is_noncanonical(self, *args, **kwargs):
        """
        Returns dictionary
            dedup_type -> boolean tensor indicating whether the tokens are non-canonical under this dedup_typ
        """
        return None # not defined for all mappings


class IdentityMapping(VocabMapping):
    """
    Identity mapping, does nothing
    """
    def __init__(self, original_vocab_size):
        self._vocab_size = original_vocab_size
    
    def __call__(self, x):
        return x
    
    def deduplicate_labels(self, labels):
        return labels

    def deduplicate_logits(self, logits):
        return logits
    
    @property
    def input_vocab_size(self):
        return self._vocab_size

    @property
    def output_vocab_size(self):
        return self._vocab_size

    @property
    def dedup_vocab_size(self):
        return self._vocab_size

    def __str__(self):
        return "identity"


class DuplicationMapping(VocabMapping):
    """
    Duplicates the vocabulary, maps tokens to their duplicates with probability 0.5
    """
    def __init__(self, original_vocab_size, frac_duplicated, p_duplicate, seed):
        self.original_vocab_size = original_vocab_size
        
        # randomly select a fraction of subword ids that are duplicated
        n_duplicated = int(frac_duplicated * original_vocab_size)
        is_duplicated = torch.zeros(original_vocab_size, dtype=torch.bool)
        all_ids = list(range(original_vocab_size))
        random.Random(seed).shuffle(all_ids)
        for i in all_ids[:n_duplicated]:
            is_duplicated[i] = True

        self.n_duplicated = n_duplicated
        self.is_duplicated = is_duplicated
        self.p_duplicate = p_duplicate
        self.seed = seed
    
    def __call__(self, x):
        do_duplicate = torch.rand_like(x, dtype=torch.float32) < self.p_duplicate
        self.is_duplicated = self.is_duplicated.to(x.device)
        do_duplicate &= self.is_duplicated[x]
        return torch.where(do_duplicate, x + self.original_vocab_size, x)
    
    def deduplicate_logits(self, logits):
        bsz, seqlen, n_classes = logits.shape
        assert n_classes == 2 * self.original_vocab_size

        # add up logits for same tokens:
        # new_logit = log(exp(logit_original) + exp(logit_duplicate))
        logits = logits.reshape(bsz, seqlen, 2, self.original_vocab_size)
        dedup_logits = torch.logsumexp(logits, dim=2)

        return dedup_logits

    def deduplicate_labels(self, labels):
        return labels % self.original_vocab_size
    
    @property
    def input_vocab_size(self):
        return self.original_vocab_size

    @property
    def output_vocab_size(self):
        return 2 * self.original_vocab_size

    @property
    def dedup_vocab_size(self):
        return self.original_vocab_size

    def __str__(self):
        return f"duplication_{self.original_vocab_size}_{self.n_duplicated}_{self.seed}_{self.p_duplicate}"


DEDUP_TRANSFORMS = {
    "whitespace": lambda x: x.strip(),
    "lower": lambda x: x.lower(),
    "plural": lambda x: x if len(x.strip()) < 4 else (x if x[-1] != "s" else x[:-1]),
    "all": lambda x: (x if len(x.strip()) < 4 else (x if x[-1] != "s" else x[:-1])).lower().strip()
}


class DeduplicationMapping(VocabMapping):
    """
    Deduplicates the vocabulary by merging near duplicate subwords
    """
    def __init__(self, sp, dedup_type):
        """
        Args:
            sp: the sentencepiece tokenizer
            dedup_type: one of "whitespace", "lower", "plural", "all"
        """
        if dedup_type not in DEDUP_TRANSFORMS:
            raise ValueError(f"Unknown deduplication type: {dedup_type}")
        self.sp = sp
        self.dedup_type = dedup_type
        self.transform = DEDUP_TRANSFORMS[dedup_type]
        self.original_vocab_size = sp.get_piece_size()
    
        # get subwords and their canonical forms (leave bytes marked by <> untouched)
        all_subwords = [sp.id_to_piece(i) if "<" in sp.id_to_piece(i) else sp.decode(i) for i in range(self.original_vocab_size)]
        all_subwords_canonical = [self.transform(subword) for subword in all_subwords]

        # assign each canonical form an id
        canonical_to_id = dict()
        for i, (sw, sw_canonical) in enumerate(zip(all_subwords, all_subwords_canonical)):
            if sw == sw_canonical:
                # if canonical subword is in vocabulary, use it
                canonical_to_id[sw_canonical] = i
            elif sw_canonical not in canonical_to_id:
                # if canonical subword is not in vocabulary, use the id of the first item that maps to it
                canonical_to_id[sw_canonical] = i
    
        self.mapping = torch.tensor([canonical_to_id[sw_canonical] for sw_canonical in all_subwords_canonical])

        # track other properties
        if dedup_type != "all":
            self._is_noncanonical = {dedup_type: torch.tensor([sw != sw_canonical for sw, sw_canonical in zip(all_subwords, all_subwords_canonical)])}
            self._is_remapped = {dedup_type: self.mapping != torch.arange(self.original_vocab_size)}
        else:
            self._is_noncanonical, self._is_remapped = {}, {}
            for dt in DEDUP_TRANSFORMS:
                if dt == "all": continue
                dedup_mapping = DeduplicationMapping(sp, dt)
                self._is_noncanonical[dt] = dedup_mapping._is_noncanonical[dt]
                self._is_remapped[dt] = dedup_mapping._is_remapped[dt]
    
    def __call__(self, x):
        self.mapping = self.mapping.to(x.device)
        return self.mapping[x]
    
    @property
    def input_vocab_size(self):
        return self.original_vocab_size

    @property
    def output_vocab_size(self):
        # always just the original size, even if some ids might not be used
        return self.original_vocab_size

    @property
    def dedup_vocab_size(self):
        # always just the original size, even if some ids might not be used
        return self.original_vocab_size
    
    def is_noncanonical(self, x, definition="remapped"):
        """
        Returns dictionary
            dedup_type -> boolean tensor indicating whether the tokens are non-canonical under this dedup_type

        Args:
            x: token ids
            definition: one of "remapped" or "noncanonical"
                "remapped": tokens that are actually remapped
                "noncanonical": tokens that are remapped and that would be remapped if their canonical form was in the vocab
        """
        masks = self._is_remapped if definition == "remapped" else self._is_noncanonical
        for dt in masks:
            masks[dt] = masks[dt].to(x.device)
        return {dt: mask[x] for dt, mask in masks.items()}

    def deduplicate_logits(self, logits):
        # compute manual logsumexp for logits

        bsz, seqlen, n_classes = logits.shape
        assert n_classes == self.output_vocab_size == self.original_vocab_size

        # exponentiate logits (with trick for numerical stability)
        max_logit = torch.max(logits)
        exp_logits = (logits - max_logit).double().exp()

        # add up logits of duplicates that are mapped together
        combined_exp_logits = torch.zeros_like(logits, dtype=torch.double)
        mapping = self.mapping.view(1, 1, n_classes).repeat(bsz, seqlen, 1).to(logits.device)
        combined_exp_logits.scatter_add_(2, mapping, exp_logits)

        # revert exponentiation and trick
        dedup_logits = (torch.log(combined_exp_logits) + max_logit).float()

        return dedup_logits

    def deduplicate_labels(self, labels):
        # just map labels
        return self(labels)
    
    def __str__(self):
        return f"deduplication_{self.original_vocab_size}_[{self.dedup_type}]"


class HalfDeduplicationMapping(DeduplicationMapping):
    """
    Partially deduplicates the vocabulary by merging half of all near duplicate subwords
    """
    def __init__(self, sp, dedup_type):
        """
        Args:
            sp: the sentencepiece tokenizer
            dedup_type: one of "whitespace", "lower", "plural", "all"
            seed: seed to use for selecting which subwords to deduplicate
        """
        super().__init__(sp, dedup_type)

        # randomly select half of the subwords to deduplicate
        rng = np.random.default_rng(0) # use numpy for consistency with old code
        do_deduplicate = torch.tensor(rng.random(len(self.mapping)) < 0.5).to(self.mapping.device)
        self.mapping = torch.where(do_deduplicate, self.mapping, torch.arange(self.original_vocab_size))

        self._is_noncanonical = None
        self._is_remapped = None

    def is_noncanonical(self, *args, **kwargs):
        return None # not supported for half deduplication
    
    def __str__(self):
        return f"halfdeduplication_{self.original_vocab_size}_[{self.dedup_type}]_{self.seed}"


def configure_dedup_mapping(
        sp,
        frac_duplicated,
        p_duplicate,
        dedup_type,
        seed=0,
    ):
    """
    Configures a deduplication mapping based on the config. Tracks vocab size etc.

    Args:
        sp: the sentencepiece tokeniser
        frac_duplicated: fraction of vocabulary to duplicate
        p_duplicate: probability of duplicating a token (only used for duplication)
        dedup_type: type of deduplication to apply ("whitespace", "lower", "plural", "all"), or ""/None for none
        verbose: whether to print the mapping
    """
    assert not (frac_duplicated > 0 and dedup_type), "Cannot have both duplication and deduplication"

    if frac_duplicated > 0:
        mapping = DuplicationMapping(
            sp.vocab_size(),
            frac_duplicated=frac_duplicated,
            p_duplicate=p_duplicate,
            seed=seed
        )
    elif dedup_type:
        assert p_duplicate is None
        if dedup_type.endswith("_50%"):
            dedup_type = dedup_type[:-4]
            mapping = HalfDeduplicationMapping(sp, dedup_type)
        else:
            mapping = DeduplicationMapping(sp, dedup_type)
    else:
        mapping = IdentityMapping(sp.vocab_size())

    return mapping
