# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_bdd10k_panoptic.py
# Modified by bmhung
# ------------------------------------------------------------------------------

import json
import os.path

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

BDD10K_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "unlabeled"},
    {"color": [180, 120, 120], "id": 1, "isthing": 0, "name": "dynamic"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "ego vehicle"},
    {"color": [80, 50, 50], "id": 3, "isthing": 0, "name": "ground"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "static"},
    {"color": [120, 120, 80], "id": 5, "isthing": 0, "name": "parking"},
    {"color": [140, 140, 140], "id": 6, "isthing": 0, "name": "rail track"},
    {"color": [204, 5, 255], "id": 7, "isthing": 0, "name": "road"},
    {"color": [230, 230, 230], "id": 8, "isthing": 0, "name": "sidewalk"},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "bridge"},
    {"color": [224, 5, 255], "id": 10, "isthing": 0, "name": "building"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "fence"},
    {"color": [150, 5, 61], "id": 12, "isthing": 0, "name": "garage"},
    {"color": [120, 120, 70], "id": 13, "isthing": 0, "name": "guard rail"},
    {"color": [8, 255, 51], "id": 14, "isthing": 0, "name": "tunnel"},
    {"color": [255, 6, 82], "id": 15, "isthing": 0, "name": "wall"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "banner"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "billboard"},
    {"color": [255, 51, 7], "id": 18, "isthing": 0, "name": "lane divider"},
    {"color": [204, 70, 3], "id": 19, "isthing": 0, "name": "parking sign"},
    {"color": [0, 102, 200], "id": 20, "isthing": 0, "name": "pole"},
    {"color": [61, 230, 250], "id": 21, "isthing": 0, "name": "polegroup"},
    {"color": [255, 6, 51], "id": 22, "isthing": 0, "name": "street light"},
    {"color": [11, 102, 255], "id": 23, "isthing": 0, "name": "traffic cone"},
    {"color": [255, 7, 71], "id": 24, "isthing": 0, "name": "traffic device"},
    {"color": [255, 9, 224], "id": 25, "isthing": 0, "name": "traffic light"},
    {"color": [9, 7, 230], "id": 26, "isthing": 0, "name": "traffic sign"},
    {"color": [220, 220, 220], "id": 27, "isthing": 0, "name": "traffic sign frame"},
    {"color": [255, 9, 92], "id": 28, "isthing": 0, "name": "terrain"},
    {"color": [112, 9, 255], "id": 29, "isthing": 0, "name": "vegetation"},
    {"color": [8, 255, 214], "id": 30, "isthing": 0, "name": "sky"},
    {"color": [7, 255, 224], "id": 31, "isthing": 1, "name": "person"},
    {"color": [255, 184, 6], "id": 32, "isthing": 1, "name": "rider"},
    {"color": [10, 255, 71], "id": 33, "isthing": 1, "name": "bicycle"},
    {"color": [255, 41, 10], "id": 34, "isthing": 1, "name": "bus"},
    {"color": [7, 255, 255], "id": 35, "isthing": 1, "name": "car"},
    {"color": [224, 255, 8], "id": 36, "isthing": 1, "name": "caravan"},
    {"color": [102, 8, 255], "id": 37, "isthing": 1, "name": "motorcycle"},
    {"color": [255, 61, 6], "id": 38, "isthing": 1, "name": "trailer"},
    {"color": [255, 194, 7], "id": 39, "isthing": 1, "name": "train"},
    {"color": [255, 122, 8], "id": 40, "isthing": 1, "name": "truck"},
]

BDD10K_COLORS = [k["color"] for k in BDD10K_CATEGORIES]

MetadataCatalog.get("bdd10k_sem_seg_train").set(
    stuff_colors=BDD10K_COLORS[:],
)

MetadataCatalog.get("bdd10k_sem_seg_val").set(
    stuff_colors=BDD10K_COLORS[:],
)

def load_bdd10k_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    def _convert_category_id(segment_info, meta):
        if int(segment_info["id"].split("-")[2]) in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["isthing"] = True
        else:
            segment_info["isthing"] = False
        segment_info["category_id"] = segment_info["category"]
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info:
        image_file = os.path.join(image_dir,ann["name"])
        label_file = os.path.join(gt_dir,os.path.splitext(ann["name"])[0] + '.png')
        sem_label_file = os.path.join(semseg_dir, os.path.splitext(ann["name"])[0] + '.png')
        segments_info = [_convert_category_id(x,meta) for x in ann["labels"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": ann["name"].split('.')[0],
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret

def register_bdd10k_panoptic(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None,
):
    """
        Register a "standard" version of bdd10k panoptic segmentation dataset named `name`.
        The dictionaries in this registered dataset follows detectron2's standard format.
        Hence it's called "standard".
        Args:
            name (str): the name that identifies a dataset,
                e.g. "bdd10k_panoptic_train"
            metadata (dict): extra metadata associated with this dataset.
            image_root (str): directory which contains all the images
            panoptic_root (str): directory which contains panoptic annotation images in COCO format
            panoptic_json (str): path to the json panoptic annotation file in COCO format
            sem_seg_root (none): not used, to be consistent with
                `register_coco_panoptic_separated`.
            instances_json (str): path to the json instance annotation file
        """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda : load_bdd10k_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="bdd10k_panoptic_seg",
        # ignore_label=255,
        # label_divisor=1000,
        **metadata,
    )
    
_PREDEFINED_SPLITS_BDD10K_PANOPTIC = {
    "bdd10k_panoptic_train": (
        "datasets/bdd10k/images/train",
        "datasets/bdd10k/labels/pan_seg/bitmasks/train",
        "datasets/bdd10k/labels/pan_seg/polygons/pan_seg_train.json",
        "datasets/bdd10k/labels/sem_seg/masks/train",
        "bdd10k/bdd10k_instance_train.json",
    ),
    "bdd10k_panoptic_val": (
        "datasets/bdd10k/images/val",
        "datasets/bdd10k/labels/pan_seg/bitmasks/val",
        "datasets/bdd10k/labels/pan_seg/polygons/pan_seg_val.json",
        "datasets/bdd10k/labels/sem_seg/masks/val",
        "bdd10k/bdd10k_instance_val.json",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in BDD10K_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in BDD10K_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in BDD10K_CATEGORIES]
    stuff_colors = [k["color"] for k in BDD10K_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
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
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root, instance_json),
    ) in _PREDEFINED_SPLITS_BDD10K_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_bdd10k_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, instance_json),
        )

_root = os.getenv("", "datasets")
register_all_bdd10k_panoptic(_root)
