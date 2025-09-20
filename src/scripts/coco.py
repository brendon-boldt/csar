from pathlib import Path
import argparse
import json
import random
import typing

from ..common import ImageMap, ImageData, Dataset, AnnotatedUtterance
from ..util import write_input_data, load_input_data
from .util import run
from ..induction import CsarPipeline

COCO_DATASET_PATH = Path("data/coco-processed.json")


def build_dataset() -> None:
    try:
        with open("data/coco/instances_val2014.json") as fo:
            data_instances = json.load(fo)
        with open("data/coco/captions_val2014.json") as fo:
            data_captions = json.load(fo)

        categories = {x["id"]: x["name"] for x in data_instances["categories"]}

        image_map: ImageMap = {}

        for annotation in data_instances["annotations"]:
            cat = categories[annotation["category_id"]]
            image_id = annotation["image_id"]
            image_data: ImageData = image_map.get(
                image_id, {"captions": [], "instances": []}
            )
            image_data["instances"].append(cat)
            image_map[image_id] = image_data

        for annotation in data_captions["annotations"]:
            image_id = annotation["image_id"]
            image_data = image_map.get(image_id, {"captions": [], "instances": []})
            image_data["captions"].append(annotation["caption"])
            image_map[image_id] = image_data

        dataset: list[tuple[list, list]] = []
        accpetable_chars = "abcdefghijklmnopqrstuvwxyz "
        for image_data in image_map.values():
            for caption in image_data["captions"]:
                caption = caption.lower()
                caption = "".join(c for c in caption if c in accpetable_chars)
                utterance = caption.split(" ")
                if len(utterance) == 0 or len(image_data["instances"]) == 0:
                    continue
                dataset.append(
                    (
                        utterance,
                        image_data["instances"],
                    )
                )
        process_dataset(typing.cast(Dataset, dataset))

    except Exception:
        breakpoint()


def process_dataset(raw_dataset: Dataset) -> None:
    def process(au: AnnotatedUtterance) -> AnnotatedUtterance:
        au["semantics"] = list(set(au["semantics"]))
        utt = au["utterance"]
        utt = utt.lower()
        accpetable_chars = "abcdefghijklmnopqrstuvwxyz "
        utt = "".join(c for c in utt if c in accpetable_chars)
        au["utterance"] = utt
        return au

    dataset: Dataset = [process(x) for x in raw_dataset]

    COCO_DATASET_PATH.parents[0].mkdir(exist_ok=True, parents=True)
    with COCO_DATASET_PATH.open("w") as fo:
        json.dump(dataset, fo)


def main(
    max_lines: int | None, max_inventory_size: int | None, write_output: bool
) -> None:
    input_path = COCO_DATASET_PATH
    random.seed(0)
    observations = load_input_data(input_path)
    random.shuffle(observations)

    pipeline = CsarPipeline(
        observations[: max_lines or 20_000],
        max_ngram=3,
        max_semcomps=2,
        trim_threshold=10,
        search_best_sub=False,
        max_inventory_size=max_inventory_size or 300,
        vocab_size=100_000,
        token_vocab_size=1000,
    )

    run("coco", pipeline, write_output=write_output)
