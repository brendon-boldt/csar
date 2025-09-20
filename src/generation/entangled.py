# type: ignore
from pathlib import Path
import json

import numpy as np

# import pulp
import cvxpy as cp
from tqdm import tqdm

from ..common import Dataset

OUT_DIR = Path("./data/synth/entangled")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cvxpy() -> None:
    n_attrs = 4
    n_vals = 4
    vocab_size = 24
    vocab_emb_dim = 20

    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 1, (vocab_size, vocab_emb_dim))
    mapper = rng.normal(0, 1, (vocab_emb_dim, n_attrs * n_vals))
    # embeddings = rng.normal(0, 1, (vocab_size, n_vals * n_attrs))
    # mapper = np.eye(n_vals * n_attrs)

    all_obs = (
        np.repeat(np.arange(n_vals**n_attrs), n_attrs, axis=0).reshape(-1, n_attrs)
        // (n_vals ** np.arange(n_attrs))
        % n_vals
    )

    def obs2vec(x: np.ndarray) -> np.ndarray:
        return np.eye(n_vals)[x].reshape(*x.shape[:-1], -1)

    all_semantics = obs2vec(all_obs)

    def word2vec(s: str) -> np.ndarray:
        return embeddings[[ord(x) - 65 for x in s]].sum(-2)

    def test_word(word: str) -> np.ndarray:
        vec = word2vec(word)
        min_obj = ((all_semantics - (vec @ mapper)) ** 2).sum(-1).argmin()
        return all_obs[min_obj]

    def make_word(x: np.ndarray) -> str:
        x = x.round().astype(int)
        return "".join(chr(i + ord("A")) * x[i] for i in range(len(x)))

    def find_best_words() -> None:
        records: Dataset = []
        for idx, obs in tqdm(list(enumerate(all_obs))):
            vocab = cp.Variable(vocab_size, integer=True)
            mapped = vocab @ embeddings @ mapper
            # objective = cp.Minimize(cp.sum_squares(expr))
            objective = cp.Minimize(vocab.sum())
            constraints: list = [
                # cp.sum_squares(obs_vec - mapped) <= cp.sum_squares(other_obs - mapped).min(),
                *(
                    mapped[attr_idx * n_vals + val]
                    >= 0.1 + mapped[attr_idx * n_vals + val_idx]
                    for attr_idx, val in enumerate(obs)
                    for val_idx in range(n_vals)
                    if val_idx != val
                ),
                # vocab.sum() <= 10,
                vocab.sum() >= 1,
                vocab.min() >= 0,
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            # print(obs, vocab.value.astype(int), prob.status, prob.value)
            assert vocab.value is not None, prob.status
            word = make_word(vocab.value)
            inv_obs = test_word(word)
            correct = (obs == inv_obs).all()
            np.set_printoptions(precision=2)
            # print(f"{obs} {word:>10s} {inv_obs} {res_str}")
            # if res_str == "x":
            #     print(vocab.value, vocab.value.sum())
            #     print(vocab.value @ embeddings @ mapper)
            #     print(obs_vec)

            if correct:
                records.append(
                    {
                        "utterance": " ".join(word),
                        "semantics": [f"{i}-{v}" for i, v in enumerate(obs)],
                    }
                )

        with (OUT_DIR / "4x4.json").open("w") as fo:
            json.dump(records, fo)

            # vars = [pulp.LpVariable(chr(i + 65), 0, 10) for i in range(vocab_size)]

    find_best_words()


def main() -> None:
    cvxpy()


if __name__ == "__main__":
    main()
