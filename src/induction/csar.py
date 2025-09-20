from typing import Iterable, Any, TypeVar
import typing
from collections import Counter

import numpy as np
import scipy.sparse
import numba  # type: ignore
from tqdm import tqdm

from ..common import Observation, Morpheme, Token, sfrozenset, InductionMorpheme
from .common import FMSetPair, DatasetRecord
from .util import get_ngrams, powerset
from ..logutils import logging
from .pipeline import Pipeline

logger = logging.getLogger(__name__)

type sparray = scipy.sparse.sparray
swindow = np.lib.stride_tricks.sliding_window_view

logging.getLogger("numba.core").setLevel(
    max(logging.INFO, logging.getLogger().getEffectiveLevel())
)

T = TypeVar("T")

# TODO See if we can optimize deocding by selecting multiple elements at once
# if they do not overlap.  I am skeptical that we can easily enough disentangle
# the decoding process.  But it is worth thinking about.


class SearchItem:
    def __init__(
        self,
        form_idx: int,
        meaning_idx: int,
        pa_idx: int,
        *,
        extractor: "Extractor",
    ) -> None:
        self.form_idx = form_idx
        self.meaning_idx = meaning_idx
        self.fm_idx = (form_idx, meaning_idx)
        self.pa_idx = pa_idx
        self.extractor = extractor
        self.corpus = extractor.corpus
        self.possible_expansions: list[SearchItem] = []
        self.applications: list[tuple[DatasetRecord, int]] = []
        self.induced_prevalence: float | None = None

    def increment_form_use(self) -> None:
        new_count = self.extractor.priority_array[2, self.pa_idx] + 1
        update_mask = self.extractor.fm_array[0] == self.form_idx
        self.extractor.priority_array[2, update_mask] = new_count

    def to_induction_morpheme(self) -> InductionMorpheme:
        cr = self.corpus
        ex = self.extractor
        if self.induced_prevalence is None:
            raise ValueError()
        return InductionMorpheme(
            form=cr.tids2form(cr.form_rindexer[self.form_idx]),
            meaning=cr.tids2meaning(cr.meaning_rindexer[self.meaning_idx]),
            induced_weight=ex.priority_array[0, self.pa_idx],
            initial_weight=ex.priority_array[1, self.pa_idx],
            prevalence=self.induced_prevalence,
        )

    def __str__(self) -> str:
        crp = self.corpus
        form = crp.tids2form(crp.form_rindexer[self.form_idx])
        meaning = crp.tids2meaning(crp.meaning_rindexer[self.meaning_idx])
        return f"{form} {set(meaning)} {self.extractor.priority_array[:, self.pa_idx]}"


@numba.jit(nopython=True)
def _update_form_occs(
    *,
    oi: np.int64,
    foccs_coords: np.ndarray,
    fr_flat_data: np.ndarray,
    fr_flat_idxr: np.ndarray,
    form_occs_data: np.ndarray,
    record_form_tids: np.ndarray,
) -> None:
    update_mask = (foccs_coords[0] == oi) & (form_occs_data > 0)
    for data_idx in update_mask.nonzero()[0]:
        fi2 = foccs_coords[1][data_idx]
        f_tids = fr_flat_data[fr_flat_idxr[fi2] : fr_flat_idxr[fi2 + 1]]
        count = 0
        L = len(f_tids)
        for i in range(record_form_tids.shape[0] - L + 1):
            good = 1
            for j in range(L):
                if record_form_tids[i + j] != f_tids[j]:
                    good = 0
                    break
            count += good
        form_occs_data[data_idx] = count


NULL_FORM = -999999


def to_obs(x: tuple) -> Observation:
    return tuple(x[0]), sfrozenset(x[1])


class Corpus:
    def __init__(
        self,
        observations: Iterable[Observation],
        *,
        max_ngram: int,
        max_semcomps: int,
        trim_threshold: int,
        ngram_semantics: bool,
        vocab_size: int | None,
        token_vocab_size: int | None,
    ) -> None:

        self.observations = list(observations)
        self.obs_indexer: dict[Observation, int] = {}
        _obs_set = set(to_obs(o) for o in self.observations)
        self.obs_indexer = {k: i for i, k in enumerate(sorted(_obs_set))}
        self.obs_occs = np.zeros(len(_obs_set), dtype=np.int64)
        for o in observations:
            self.obs_occs[self.obs_indexer[to_obs(o)]] += 1
        self.max_semcomps = max_semcomps
        self.max_ngram = max_ngram
        self.trim_threshold = trim_threshold
        self.vocab_size = vocab_size
        self.token_vocab_size = vocab_size
        self.ngram_semantics = ngram_semantics

    def _make_fm_candidates(self) -> list[FMSetPair]:
        form_token_counter: Counter[str | int] = Counter()
        meaning_token_counter: Counter[str | int] = Counter()

        logger.info("Building form and meaning sets")
        for utterance, meaning in self.obs_indexer.keys():
            form_token_counter.update(utterance)
            meaning_token_counter.update(meaning)

        form_tokens_to_keep = [
            x for x, _ in form_token_counter.most_common()[: self.token_vocab_size]
        ]
        meaning_tokens_to_keep = [
            x for x, _ in meaning_token_counter.most_common()[: self.token_vocab_size]
        ]

        form_tokens_to_keep.append(NULL_FORM)

        # TODO Can form_occs be boolean again if we are not using the numbers?
        self.form_token_indexer = {
            v: i for i, v in enumerate(sorted(form_tokens_to_keep, key=str))
        }
        self.form_token_rindexer = np.fromiter(
            self.form_token_indexer.keys(), dtype="O"
        )
        self.meaning_token_indexer = {
            v: i for i, v in enumerate(sorted(meaning_tokens_to_keep, key=str))
        }
        self.meaning_token_rindexer = np.fromiter(
            self.meaning_token_indexer.keys(), dtype="O"
        )

        form_counter: Counter[tuple[int, ...]] = Counter()
        meaning_counter: Counter[sfrozenset[int]] = Counter()

        self.dataset: list[DatasetRecord] = []
        set_pairs: list[FMSetPair] = []
        logger.info("Building form-meaning set pairs")
        for utterance, meaning in self.obs_indexer.keys():
            # TODO Fix -1 magic number
            form_tokens = [self.form_token_indexer.get(x, -1) for x in utterance]
            ngrams = get_ngrams(form_tokens, self.max_ngram)
            meaning_tokens = [
                self.meaning_token_indexer[x]
                for x in meaning
                if x in self.meaning_token_indexer
            ]
            if self.ngram_semantics:
                semantics = sfrozenset(
                    [
                        sfrozenset(x)
                        for x in get_ngrams(meaning_tokens, self.max_semcomps)
                    ]
                )
            else:
                semantics = powerset(meaning_tokens, self.max_semcomps)
            meaning_counter.update(semantics)
            form_counter.update(ngrams)
            set_pairs.append((ngrams, semantics))
            assert all(None not in x for x in ngrams)
            form_tokens_np = np.array(form_tokens)
            rec = DatasetRecord(
                form_tids=form_tokens_np,
                form_tids_original=form_tokens_np.copy(),
                meaning_tids=sfrozenset(meaning_tokens),
            )
            self.dataset.append(rec)

        def form_key(x: tuple[str, ...]) -> Any:
            return len(x), x

        forms_to_keep = [x for x, _ in form_counter.most_common()[: self.vocab_size]]
        meanings_to_keep = [
            x for x, _ in meaning_counter.most_common()[: self.vocab_size]
        ]
        forms_to_keep = list(form_counter.keys())
        meanings_to_keep = list(meaning_counter.keys())

        self._null_form = (self.form_token_indexer[NULL_FORM],)
        forms_to_keep.append(self._null_form)

        self.form_indexer = {v: i for i, v in enumerate(sorted(forms_to_keep))}
        self.form_rindexer = np.fromiter(self.form_indexer.keys(), dtype="O")
        self.fr_flat_data = np.concat(self.form_rindexer)
        self.fr_flat_idxr = np.cumsum([0] + [len(x) for x in self.form_rindexer])
        self.meaning_indexer = {v: i for i, v in enumerate(sorted(meanings_to_keep))}
        self.meaning_rindexer = np.fromiter(self.meaning_indexer.keys(), dtype="O")

        return set_pairs

    def count_cooccurrences(self, use_null_form: bool = False) -> None:
        set_pairs = self._make_fm_candidates()

        n_forms = len(self.form_indexer)
        n_meanings = len(self.meaning_indexer)

        n_observations = len(set_pairs)
        meaning_occs = scipy.sparse.lil_array(
            (n_observations, n_meanings), dtype=np.int64
        )
        form_occs = scipy.sparse.lil_array((n_observations, n_forms), dtype=np.int64)

        logger.info("Counting occurrences")
        # TODO Should this process be unified with the later updating of forms
        # as they are subtracted?
        for i, p in sorted(list(enumerate(set_pairs)), key=lambda x: x[1]):
            form_row = np.zeros(n_forms, dtype=np.int64)
            meaning_row = np.zeros(n_meanings, dtype=np.int64)
            for x in p[0]:
                j = self.form_indexer.get(x, None)
                if j is not None:
                    form_row[j] += 1
            fr_idxs = form_row.nonzero()[0]
            form_occs.rows[i] = fr_idxs.tolist()
            form_occs.data[i] = form_row[fr_idxs].tolist()
            for y in p[1]:
                k = self.meaning_indexer.get(y, None)
                if k is not None:
                    meaning_row[k] += 1
            mr_idxs = meaning_row.nonzero()[0]
            meaning_occs.rows[i] = mr_idxs.tolist()
            meaning_occs.data[i] = meaning_row[mr_idxs].tolist()
        if use_null_form:
            form_occs[:, self.form_indexer[self._null_form]] = 999_999

        form_occs = form_occs.tocoo()
        meaning_occs = meaning_occs.tocoo()

        form_sums = form_occs.sum(0)
        form_trim_mask = form_sums <= self.trim_threshold
        form_occs.data[form_trim_mask[form_occs.coords[1]]] = 0

        meaning_trim_mask = meaning_occs.sum(0) <= self.trim_threshold
        meaning_occs.data[meaning_trim_mask[meaning_occs.coords[1]]] = 0

        self.form_occs = form_occs
        self.meaning_occs = meaning_occs

    def tids2meaning(self, tids: Iterable[int]) -> sfrozenset[Token]:
        return sfrozenset({self.meaning_token_rindexer[x] for x in tids})

    def tids2form(self, tids: Iterable[int]) -> tuple[Token, ...]:
        return tuple(self.form_token_rindexer[x] for x in tids)


WeightMethod = typing.Literal["mi", "app", "pmim", "jp"]
WeightFunc = typing.Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _w_mutual_information(
    pfm: np.ndarray, pf: np.ndarray, pm: np.ndarray
) -> np.ndarray:
    def f(xy, x, y):
        x_times_y = x * y
        _zl = np.zeros_like
        log_in = np.divide(xy, x_times_y, where=x_times_y > 0, out=np.zeros_like(xy))
        log_out = np.log2(log_in, where=log_in > 0, out=np.zeros_like(log_in))
        return xy * log_out

    return -(
        f(pfm, pf, pm)
        + f(pm - pfm, 1 - pf, pm)
        + f(pf - pfm, pf, 1 - pm)
        + f(1 - pf - pm + pfm, 1 - pf, 1 - pm)
    )


def _w_applicability(pfm: np.ndarray, pf: np.ndarray, pm: np.ndarray) -> np.ndarray:
    def f(xy, x):
        return xy**2 / (x + 1e-10)

    return -(1 + (f(pfm, pf) - f(pm - pfm, 1 - pf) - f(pf - pfm, pf)))


def _w_pmi_mass(pfm: np.ndarray, pf: np.ndarray, pm: np.ndarray) -> np.ndarray:
    return -pfm * np.log2(1e-10 + pfm / (pf * pm + 1e-10))


def _w_joint_prob(pfm: np.ndarray, pf: np.ndarray, pm: np.ndarray) -> np.ndarray:
    return -pfm


_weight_funcs: dict[WeightMethod, WeightFunc] = {
    "mi": _w_mutual_information,
    "app": _w_applicability,
    "pmim": _w_pmi_mass,
    "jp": _w_joint_prob,
}


class Extractor:
    def __init__(
        self,
        corpus: Corpus,
        *,
        trim_threshold: int,
        search_best_sub: bool,
        max_inventory_size: int,
        show_progress: bool,
        weight_method: WeightMethod,
    ) -> None:
        self.corpus = corpus
        self.use_null_form = False
        self.trim_threshold = trim_threshold
        self.search_best_sub = search_best_sub
        self.max_inventory_size = max_inventory_size
        self.show_progress = show_progress
        self.weight_method = weight_method
        self._weight_func = _weight_funcs[self.weight_method]

    def _compute_prios(self) -> tuple[sparray, sparray]:
        """Compute the prio of all form meaning pairs (lower is better)."""
        n_obs = self._obs_occs_total
        focc = self.form_occs.copy()
        np.clip(focc.data, max=1, out=focc.data)  # type: ignore[call-overload]
        focc *= self.corpus.obs_occs[:, None]
        uw_mocc = self.meaning_occs.copy()
        mocc = uw_mocc * self.corpus.obs_occs[:, None]

        p_fm_sparray = (focc.T @ uw_mocc) / n_obs
        thresh = self.trim_threshold / n_obs

        if thresh > 0:
            p_fm_sparray.data[p_fm_sparray.data <= thresh] = 0

        self._n_uniq_coocs = p_fm_sparray.size
        fm_coords = p_fm_sparray.tocoo().coords

        p_f_sparray = focc.sum(0) / n_obs
        p_m_sparray = mocc.sum(0) / n_obs
        p_f = p_f_sparray[fm_coords[0]]
        p_m = p_m_sparray[fm_coords[1]]
        p_fm = p_fm_sparray.data

        prios = self._weight_func(p_fm, p_f, p_m)
        prios = np.where(p_fm < 1e-10, 0, prios)
        prios_sparray = scipy.sparse.csr_array(
            (prios, p_fm_sparray.indices, p_fm_sparray.indptr),
            shape=p_fm_sparray.shape,
        )
        return prios_sparray, p_fm_sparray

    def _get_best_substitution(
        self,
        *,
        fo_lil: sparray,
        mo_lil: sparray,
        sorted_fm_idxs: np.ndarray,
        occ_idx: int,
        form_tids: np.ndarray,
        matches: np.ndarray,
        match_idxs: np.ndarray,
    ) -> int:
        # TODO parameterize
        if self.search_best_sub is False:
            return match_idxs[occ_idx % len(match_idxs)]

        record = self.corpus.dataset[occ_idx]
        # TODO Get rid of this magic number
        tmp_form_mask = record.form_tids != -1
        remaining_meanings = np.zeros(mo_lil.shape[1], dtype=np.int64)
        remaining_meanings[mo_lil.rows[occ_idx]] = mo_lil.data[occ_idx]
        meaning_mask = remaining_meanings[sorted_fm_idxs[1]]
        remaining_forms = np.zeros(fo_lil.shape[1], dtype=np.int64)
        remaining_forms[fo_lil.rows[occ_idx]] = fo_lil.data[occ_idx]
        form_mask = remaining_forms[sorted_fm_idxs[0]] > 0

        mask = meaning_mask & form_mask
        for fi2, mi2 in sorted_fm_idxs.T[mask.nonzero()]:
            form2 = self.corpus.form_rindexer[fi2]
            if len(form2) > len(tmp_form_mask):
                continue
            swmask = swindow(tmp_form_mask, len(form2))
            swtids = swindow(record.form_tids, len(form2))
            raw_update_mask = ((swtids == form2) & swmask).all(-1)
            if not raw_update_mask.any():
                continue
            update_mask = np.convolve(raw_update_mask, np.ones(len(form2), dtype=bool))
            new_tmp_form_mask = tmp_form_mask & ~update_mask

            remaining_matches = matches & swindow(
                new_tmp_form_mask, len(form_tids)
            ).all(-1)
            match remaining_matches.sum():
                case 0:
                    pass  # Continue on to further forms to break ties.
                case 1:
                    sub_idx = remaining_matches.nonzero()[0][0]
                    logger.debug(f"  selecting {sub_idx} from {match_idxs}")
                    return sub_idx
                case _:
                    tmp_form_mask = new_tmp_form_mask
        # We have no other way to break the tie
        # return match_idxs[0]
        return match_idxs[occ_idx % len(match_idxs)]

    def _handle_pair_select(self, search_item: SearchItem, best_item_idx: int) -> None:
        logger.debug(f"Best item is {search_item}")
        # self.selected_pairs[*search_item.fm_idx] = True
        self.induction_morphemes.append(search_item.to_induction_morpheme())
        self.priority_array[0, best_item_idx] = 0

        moccs_coords = self.meaning_occs.tocoo().coords
        search_item_meaning = self.corpus.meaning_rindexer[search_item.meaning_idx]
        target_meanings = np.array(
            [
                bool(search_item_meaning & self.corpus.meaning_rindexer[mi])
                for mi in moccs_coords[1]
            ]
        )

        target_observations = (
            self.form_occs[:, search_item.form_idx].todense()[moccs_coords[0]] > 0
        )
        sorted_fm_idxs: np.ndarray | None = None

        foccs_coords = self.form_occs.tocoo().coords

        obs_to_update = (
            self.meaning_occs[:, search_item.meaning_idx]
            * (self.form_occs[:, search_item.form_idx] > 0)
        ).nonzero()[0]
        if (
            search_item.form_idx
            == self.corpus.form_indexer[self.corpus.form_token_indexer[NULL_FORM],]
        ):
            obs_to_update = []

        self.meaning_occs.data[target_meanings & target_observations] -= 1
        np.clip(self.meaning_occs.data, min=0, out=self.meaning_occs.data)  # type: ignore[call-overload]

        form_tids = self.corpus.form_rindexer[search_item.form_idx]
        winsize = len(form_tids)
        oi: int
        fo_lil = self.form_occs.tolil()
        mo_lil = self.meaning_occs.tolil()
        for oi in obs_to_update:
            record = self.corpus.dataset[oi]
            candidates = swindow(record.form_tids, winsize)
            matches = (candidates == form_tids).all(-1)

            match_idxs = matches.nonzero()[0]
            # Default value if this process fails
            sub_idx = match_idxs[0]
            if matches.sum() > 1:
                if sorted_fm_idxs is None:
                    # TODO Could probably make this faster by filtering out
                    # priority values that are at zero.
                    sorted_idxs = np.lexsort(self.priority_array[::-1])
                    sorted_fm_idxs = self.fm_array[:, sorted_idxs]
                sub_idx = self._get_best_substitution(
                    fo_lil=fo_lil,
                    mo_lil=mo_lil,
                    occ_idx=oi,
                    match_idxs=match_idxs,
                    matches=matches,
                    form_tids=form_tids,
                    sorted_fm_idxs=sorted_fm_idxs,
                )
            else:
                sub_idx = match_idxs[0]

            record.form_tids[sub_idx : sub_idx + len(form_tids)] = -1
            search_item.applications.append((record, sub_idx))
            # TODO Would this be faster if we moved outside the loop and put
            # the loop inside the jitted function?
            _update_form_occs(
                oi=oi,
                record_form_tids=record.form_tids,
                form_occs_data=self.form_occs.data,
                fr_flat_idxr=self.corpus.fr_flat_idxr,
                fr_flat_data=self.corpus.fr_flat_data,
                foccs_coords=foccs_coords,
            )

        if self.use_null_form:
            # TODO very hacky to have this here
            self.form_occs[
                :, self.corpus.form_indexer[self.corpus.form_token_indexer[NULL_FORM],]
            ] = 999_999
        np.clip(self.form_occs.data, min=0, out=self.form_occs.data)  # type: ignore [call-overload]
        self.meaning_occs.eliminate_zeros()
        self.form_occs.eliminate_zeros()
        search_item.increment_form_use()

    def _refresh_priority(
        self, pa_idx_map: dict[tuple[int, int], int], raw_prios: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        new_prios, prevalences = self._compute_prios()
        changed_prios = (raw_prios - new_prios) != 0

        # Force all changed prios to be present by making them
        # non-zero; we will subtract it out below.
        new_changed_prios = (changed_prios * new_prios) + changed_prios
        ncw = new_changed_prios
        vals = ncw.data
        cols = ncw.indices
        vals_by_row = ncw.indptr[1:] - ncw.indptr[:-1]
        rows = np.repeat(np.arange(len(vals_by_row)), vals_by_row)
        for val, row, col in zip(vals, rows, cols):
            # Subtract 1 because changed_prios is a boolean array.
            # self.priority_array[0, pa_idx_map[row.item(), col.item()]] = val - 1
            new_val = val - 1
            old_val = self.priority_array[0, pa_idx_map[row.item(), col.item()]]
            self.priority_array[0, pa_idx_map[row.item(), col.item()]] = max(
                new_val, old_val
            )

        return new_prios, prevalences

    def _keep_going(self) -> bool:
        # TODO Add time budget, etc.
        meaning_sum = self.meaning_occs.data.sum()
        form_sum = self.form_occs.data.sum()
        logger.debug(f"{meaning_sum} meanings remain.")
        logger.debug(f"{form_sum} forms remain.")
        forms_and_meanings_left = (
            self.meaning_occs.data.any() and self.form_occs.data.any()
        )
        more_morphemes = len(self.induction_morphemes) < self.max_inventory_size
        return forms_and_meanings_left and more_morphemes

    N_FIELDS = 5

    def _init_priority_array(self, raw_prios: sparray) -> None:
        # TODO Provide named constants for these fields
        self.priority_array = priority_array = np.full(
            (Extractor.N_FIELDS, len(raw_prios.data)), np.nan, dtype=np.float64
        )
        self.fm_array = np.full((2, priority_array.shape[1]), -1, dtype=np.int64)
        self.search_item_list = []

        self.pa_idx_map = {}
        for idx, (fi, mi, prio) in enumerate(zip(*raw_prios.coords, raw_prios.data)):
            self.pa_idx_map[(fi, mi)] = idx
            priority_array[:, idx] = (
                prio,
                prio,
                0,
                -1 * len(self.corpus.form_rindexer[fi]),
                len(self.corpus.meaning_rindexer[mi]),
            )
            self.fm_array[:, idx] = fi, mi
            self.search_item_list.append(SearchItem(fi, mi, idx, extractor=self))
        # priority_array[3, self.fm_array[0] == self.corpus.form_indexer[self.corpus.form_token_indexer[NULL_FORM],]] = 999_999_999

    def _initialize(self) -> None:
        cr = self.corpus
        # self.overlap_mat = cr._get_overlap_mat()
        self.form_occs = cr.form_occs.tocsc(copy=True)
        self.meaning_occs = cr.meaning_occs.tocsc(copy=True)
        self.init_form_occs = cr.form_occs.tocsc(copy=True)
        self.init_meaning_occs = cr.meaning_occs.tocsc(copy=True)
        self._obs_occs_total = cr.obs_occs.sum()

    def _main_loop(self) -> None:
        logger.info("Extracting morphemes")
        raw_prios = self._compute_prios()[0].tocoo()
        self._init_priority_array(raw_prios)
        self.induction_morphemes: list[InductionMorpheme] = []

        if logger.isEnabledFor(logging.DEBUG):
            for idx in np.lexsort(self.priority_array[::-1])[:100]:
                logger.debug(f"{self.search_item_list[idx]}")

        init_prog = self._n_uniq_coocs
        pbar = tqdm(total=self._n_uniq_coocs, disable=not self.show_progress)
        while self._keep_going():
            raw_prios, prevalences = self._refresh_priority(self.pa_idx_map, raw_prios)
            min_val = raw_prio_min_val = self.priority_array[0].min()
            if min_val == 0:
                break
            working_mask = self.priority_array[0] == min_val
            for level in range(1, Extractor.N_FIELDS):
                working_mask &= (
                    self.priority_array[level]
                    == self.priority_array[level, working_mask].min()
                )
                if working_mask.sum() == 1:
                    break
            best_item_idx = working_mask.nonzero()[0][0]
            best_item = self.search_item_list[best_item_idx]
            best_item.induced_prevalence = prevalences[*best_item.fm_idx]

            same_meaning = self.fm_array[1] == self.fm_array[1, best_item_idx]
            same_prio = self.priority_array[0] == raw_prio_min_val
            # TODO Name these variables
            larger_form = self.priority_array[3] > self.priority_array[3, best_item_idx]
            best_item.possible_expansions = [
                self.search_item_list[item_idx]
                for item_idx in (same_meaning & same_prio & larger_form).nonzero()[0]
            ]
            self._handle_pair_select(best_item, best_item_idx)

            pbar.n = init_prog - self._n_uniq_coocs
            pbar.refresh()
        pbar.close()

    def run(self) -> None:
        self._initialize()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.warning("Interrupted early! Induced partial morpheme inventory...")

        self._finialize_run()

    def _finialize_run(self) -> None:
        self.morphemes = [x.to_morpheme() for x in self.induction_morphemes]


# TODO Write input validity tests.
# TODO Add options for recording position of forms.


class CsarPipeline(Pipeline):
    def __init__(
        self,
        observations: Iterable[Observation],
        *,
        max_ngram: int = 1,
        max_semcomps: int = 1,
        trim_threshold: int = 0,
        search_best_sub: bool = True,
        max_inventory_size: int | None = None,
        vocab_size: int | None = None,
        token_vocab_size: int | None = None,
        ngram_semantics: bool = False,
        show_progress: bool = False,
        weight_method: WeightMethod = "mi",
    ) -> None:
        super().__init__(observations)
        self.corpus = Corpus(
            observations,
            max_ngram=max_ngram,
            max_semcomps=max_semcomps,
            trim_threshold=trim_threshold,
            vocab_size=vocab_size,
            token_vocab_size=token_vocab_size,
            ngram_semantics=ngram_semantics,
        )
        max_inventory_size = max_inventory_size or 2**32
        self.extractor = Extractor(
            self.corpus,
            weight_method=weight_method,
            trim_threshold=trim_threshold,
            search_best_sub=search_best_sub,
            max_inventory_size=max_inventory_size,
            show_progress=show_progress,
        )

    def induce(self) -> list[Morpheme]:
        self.corpus.count_cooccurrences()
        self.extractor.run()

        return self.get_morphemes()

    def get_morphemes(self) -> list[Morpheme]:
        return self.extractor.morphemes
