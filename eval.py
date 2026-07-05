
import torch
import parser
import logging
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings("ignore")

DEFAULT_EVAL_DATASET_NAMES = [
    "tokyo247",
    "nordland",
    "sf_xl_v1",
    "sf_xl_occlusion",
    "sf_xl_night",
    "svox_night",
    "svox_overcast",
    "svox_rain",
    "svox_snow",
    "svox_sun",
]

EVAL_DATASET_GROUPS = {
    "sf_xl": ["sf_xl_v1", "sf_xl_occlusion", "sf_xl_night"],
    "sfxl": ["sf_xl_v1", "sf_xl_occlusion", "sf_xl_night"],
    "svox": ["svox_night", "svox_overcast", "svox_rain", "svox_snow", "svox_sun"],
}

EVAL_DATASET_ALIASES = {
    "tokyo": "tokyo247",
    "msls": "Msls_740",
    "msls_val": "Msls_740",
    "msls_740": "Msls_740",
    "sfxlv1": "sf_xl_v1",
    "sfxlnight": "sf_xl_night",
    "sfxlocclusion": "sf_xl_occlusion",
    "sfxlvocclusion": "sf_xl_occlusion",
}


def expand_eval_dataset_names(dataset_names):
    expanded = []
    for name in dataset_names:
        normalized = name.lower().replace("-", "_")
        compact = normalized.replace("_", "")
        canonical = EVAL_DATASET_ALIASES.get(compact, EVAL_DATASET_ALIASES.get(normalized, name))
        if canonical in EVAL_DATASET_GROUPS:
            expanded.extend(EVAL_DATASET_GROUPS[canonical])
        else:
            expanded.append(canonical)

    ordered = []
    seen = set()
    for name in expanded:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def infer_eval_split(dataset_name):
    return "val" if dataset_name.lower() == "msls_740" else "test"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.VPRNet()
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)
args.features_dim = 4096
if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
eval_dataset_names = expand_eval_dataset_names(args.eval_dataset_names or DEFAULT_EVAL_DATASET_NAMES)
test_ds_list = [
    datasets_ws.BaseDataset(
        args,
        args.eval_datasets_folder,
        name,
        infer_eval_split(name),
    )
    for name in eval_dataset_names
]
for test_ds in test_ds_list:
    logging.info(f"Test set: {test_ds}")
    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
    if isinstance(recalls_str, dict):
        for subset_name, subset_recalls in recalls_str.items():
            logging.info(f"Recalls on {subset_name}: {subset_recalls}")
    else:
        logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
