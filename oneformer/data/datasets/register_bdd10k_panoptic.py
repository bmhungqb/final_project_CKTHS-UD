# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_bdd10k_panoptic.py
# Modified by bmhung
# ------------------------------------------------------------------------------

import json
import os.path

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

BDD10K_CATEGORIES = []


def load_bdd10k_panoptic_json(json_file, image_dir, gt_dir, meta):
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        image_file = os.path.join(image_dir,os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir,ann["file_name"])
        segments_info = [_convert_category_id(x,meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret

def register_bdd10k_panoptic(
    name, image_root, panoptic_root, panoptic_json
):
    panoptic_name = name
    with PathManager.open(panoptic_json) as f:
        json_info = json.load(f)
    BDD10K_CATEGORIES = json_info["categories"]
    BDD10K_COLORS = [k["color"] for k in BDD10K_CATEGORIES]
    MetadataCatalog.get("bdd10k_sem_seg_train").set(
        stuff_colors=BDD10K_COLORS[:],
    )
    MetadataCatalog.get("bdd10k_sem_seg_val").set(
        stuff_colors=BDD10K_COLORS[:],
    )
    metadata = get_metadata()

    DatasetCatalog.register(
        panoptic_name,
        lambda : load_bdd10k_panoptic_json(
            panoptic_json, image_root, panoptic_root, metadata
        ),
    )

    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="bdd10k_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )
    
_PREDEFINED_SPLITS_BDD10K_PANOPTIC = {
    "bdd10k_panoptic_train": (
        "bdd10k/images/train",
        "bdd10k/labels/pan_seg/bitmasks/train",
        "bdd10k/labels/jsons/pan_seg_train_cocofmt.json",
    ),
    "bdd10k_panoptic_val": (
        "bdd10k/images/val",
        "bdd10k/labels/jsons/pan_seg/masks/val",
        "bdd10k/labels/jsons/pan_seg_val_cocofmt.json",
    ),
}


def get_metadata():
    meta = {}
    thing_classes = [k["name"] for k in BDD10K_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in BDD10K_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in BDD10K_CATEGORIES]
    stuff_colors = [k["color"] for k in BDD10K_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(BDD10K_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

def register_all_bdd10k_panoptic(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json),
    ) in _PREDEFINED_SPLITS_BDD10K_PANOPTIC.items():
        register_bdd10k_panoptic(
            prefix,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
        )

_root = os.getenv("", "datasets")
register_all_bdd10k_panoptic(_root)
