from pathlib import Path
import argparse
import json

from ..common import ImageMap, ImageData, Dataset, AnnotatedUtterance


COCO_DATASET_DIR = Path("data/coco-processed")
COCO_DATASET_DIR.mkdir(exist_ok=True, parents=True)


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

        dataset: Dataset = []
        for image_data in image_map.values():
            for caption in image_data["captions"]:
                dataset.append(
                    {
                        "utterance": caption,
                        "semantics": image_data["instances"],
                    }
                )

        with (COCO_DATASET_DIR / "dataset.raw.json").open("w") as fo:
            json.dump(dataset, fo)
    except Exception:
        breakpoint()


def process_dataset() -> None:
    with (COCO_DATASET_DIR / "dataset.raw.json").open() as fo:
        raw_dataset: Dataset = json.load(fo)

    def process(au: AnnotatedUtterance) -> AnnotatedUtterance:
        au["semantics"] = list(set(au["semantics"]))
        utt = au["utterance"]
        utt = utt.lower()
        accpetable_chars = "abcdefghijklmnopqrstuvwxyz "
        utt = "".join(c for c in utt if c in accpetable_chars)
        au["utterance"] = utt
        return au

    dataset: Dataset = [process(x) for x in raw_dataset]

    with (COCO_DATASET_DIR / "dataset.json").open("w") as fo:
        json.dump(dataset, fo)


def handwritten_to_dataset(args: argparse.Namespace) -> Dataset:
    dataset: Dataset = []
    with open(args.input_path) as fo:
        for line in fo:
            line = line.replace(" ", "").strip()
            form, meaning = line.split(",")
            dataset.append(
                {
                    "utterance": " ".join(form),
                    "semantics": [
                        f"{i}-{x}" for i, x in enumerate(meaning) if x != "0"
                    ],
                }
            )
    return dataset
