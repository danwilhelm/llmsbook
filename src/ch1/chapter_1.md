# Chapter 1: Solving Cryptograms

In this chapter, we will examine how natural language statistics can be encoded in transformer weights. To do this, we'll handcraft a simple transformer that solves cryptograms encoded using a **Caesar cipher** (a fixed letter rotation).

<div class="warning">
This problem was originally presented by Callum McDougall as an interpretability challenge as part of his <a href="https://github.com/callummcdougall/ARENA_3.0">ARENA bootcamp</a>. For more practice, you can explore his <a href="https://colab.research.google.com/drive/1pW1qAd52ZRf6gU-fTORjVyVKHuoLzJzH">PyTorch-trained transformer model</a> of the same problem.
</div>

## Cryptograms Intro

A **Caesar cipher** is a code where each letter has been rotated forward by a fixed number. Given rotated text (**ciphertext**), the challenge is to determine the original rotation number (and thereby recover the original **plaintext**).

> The plaintext `a bay` becomes ciphertext `d edb` with rotation 3.

Note that only the letters a-z will be rotated. For example, the space character in "a bay" is not rotated.

A transformer only accepts a fixed **vocabulary** of possible tokens. In this problem, we will allow 27 tokens -- the lowercase letters `a-z` and the space character.

---
## Python Implementation

We encourage you to code along in your own notebook! For this chapter, visit GitHub to download three Project Gutenberg text files and our minimal plotting functions in `llm_plots.py`. Place these in the same directory as your notebook, then run the notebook server from this directory to ensure it is on your PATH.

Here are the Python packages we'll use:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from dan_plots import rowplot, implot, listplot

np.set_printoptions(suppress=True)  # suppress scientific notation
```

---
Now let's implement the Caesar rotation in Python! Our model requires indexes into the vocabulary as input. So instead of storing ciphertext as a string, we will represent it as an array of rotated numbers (**rotnums**). For clarity, all strings will be assumed plaintext and all lists of numbers ciphertext.

> The ciphertext `d edb` will be stored as `[3, 26, 4, 3, 1]`, where `'a'=0` and `' '=26`.

For now, we will use classes primarily to help us organize and encapsulate constants. The class `CaesarDataset` will load text and rotate it, starting with:

```python
class CaesarDataset:
    SEQ_LEN = 32     # chars per cryptogram
    N_ROTS = 26      # total letters/rotations
    
    # VOCAB: Must begin with a-z and include space.
    VOCAB = np.array(list('abcdefghijklmnopqrstuvwxyz '))
    VOCAB_SET = set(VOCAB)                         # O(1) membership test
    VOCAB_IDX = {l:i for i,l in enumerate(VOCAB)}  # O(1) char->index

    @staticmethod
    def plaintext_to_rotnums(plaintext, rot=0):
        '''Rotate letters forward, returning rotated numbers.
        Input characters must be in VOCAB.'''
        vocab_idxs = np.array([CaesarDataset.VOCAB_IDX[c] for c in plaintext])
        return np.where(vocab_idxs < CaesarDataset.N_ROTS, 
                        (vocab_idxs + rot) % CaesarDataset.N_ROTS,
                        vocab_idxs)  # do not rotate if not a-z

    @staticmethod
    def rotnums_to_plaintext(rotnums, rot=0):
        'Rotate numbers backward, returning plaintext.
        Input numbers must be less than the VOCAB size.'
        return ''.join(CaesarDataset.VOCAB[
            np.where(rotnums < CaesarDataset.N_ROTS,
                     (rotnums - rot) % CaesarDataset.N_ROTS,
                     rotnums)])  # do not rotate if not a-z
```

<details><summary>Example Run (expand to view)</summary>

```python
rot = 5
plaintext = 'the quick brown fox jumps over the lazy dog'
rotnums = CaesarDataset.plaintext_to_rotnums(plaintext, rot)

print('plaintext:', plaintext)
print('ciphertext:', CaesarDataset.rotnums_to_plaintext(rotnums))
print('plaintext (hopefully!):',
        CaesarDataset.rotnums_to_plaintext(rotnums, rot))
```

This outputs:

```
plaintext: the quick brown fox jumps over the lazy dog
ciphertext: ymj vznhp gwtbs ktc ozrux tajw ymj qfed itl
plaintext (hopefully!): the quick brown fox jumps over the lazy dog
```
</details>
