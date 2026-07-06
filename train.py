import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import network
from loss import loss_function
from dataloaders.GSVCities import get_GSVCities

import warnings
warnings.filterwarnings("ignore")

DEFAULT_EVAL_DATASET_NAMES = [
    "pitts30k",
    "Msls_740",
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

INTERMEDIATE_EVAL_DATASET_NAMES = [
    "pitts30k",
    "Msls_740",
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


def unique_preserve_order(dataset_names):
    ordered = []
    seen = set()
    for name in dataset_names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def filter_out_sfxl(dataset_names):
    sfxl_datasets = set(EVAL_DATASET_GROUPS["sfxl"])
    return [name for name in dataset_names if name not in sfxl_datasets]


#### Initial setup: parser, logging...
args = parser.parse_arguments()
if args.resume and args.resume_author:
    raise ValueError("Use either --resume for training checkpoints or --resume_author for model-only author weights, not both.")
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading evaluation datasets from folder {args.eval_datasets_folder}")
requested_eval_dataset_names = expand_eval_dataset_names(
    args.eval_dataset_names or INTERMEDIATE_EVAL_DATASET_NAMES
)
intermediate_eval_dataset_names = unique_preserve_order(
    INTERMEDIATE_EVAL_DATASET_NAMES + requested_eval_dataset_names
)
intermediate_eval_dataset_names = [
    name for name in intermediate_eval_dataset_names if name in INTERMEDIATE_EVAL_DATASET_NAMES
]
requested_final_eval_dataset_names = expand_eval_dataset_names(args.final_eval_dataset_names)
final_eval_dataset_names = []
if requested_final_eval_dataset_names:
    final_eval_dataset_names = filter_out_sfxl(requested_final_eval_dataset_names)
    final_eval_dataset_names = unique_preserve_order(
        INTERMEDIATE_EVAL_DATASET_NAMES + final_eval_dataset_names
    )
eval_dataset_names = unique_preserve_order(
    intermediate_eval_dataset_names + final_eval_dataset_names
)
eval_datasets = {
    name: datasets_ws.BaseDataset(
        args,
        args.eval_datasets_folder,
        name,
        infer_eval_split(name),
    )
    for name in eval_dataset_names
}
for name, eval_ds in eval_datasets.items():
    logging.info(f"Eval set: {name} -> {eval_ds}")
logging.info(f"Intermediate eval datasets: {intermediate_eval_dataset_names}")
logging.info(f"Final-epoch eval datasets (excluding SF-XL): {final_eval_dataset_names}")

#### Initialize model
model = network.VPRNet(
    pretrained_foundation=args.foundation_model_path is not None,
    foundation_model_path=args.foundation_model_path,
)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

args.features_dim = 4096

# Freeze parameters except adapter
for name, param in model.module.backbone.named_parameters():
    if "adapter" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# initialize Adapter
for n, m in model.named_modules():
    if 'adapter' in n:
        for n2, m2 in m.named_modules():
            if 'D_fc2' in n2:
                if isinstance(m2, nn.Linear):
                    nn.init.constant_(m2.weight, 0.)
                    nn.init.constant_(m2.bias, 0.)

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#### Resume model, optimizer, and other training parameters
if args.resume:
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch_num = checkpoint["epoch_num"] + 1
    best_pitts_r1 = checkpoint.get("best_pitts_r1", checkpoint.get("pitts_r1", 0.0))
    best_msls_r1 = checkpoint.get("best_msls_r1", checkpoint.get("msls_r1", 0.0))
    not_improved_num = checkpoint.get("not_improved_num", 0)
    logging.info(
        f"Resuming from epoch {start_epoch_num}: "
        f"best selection tuple (Pitts30k, MSLS)=({best_pitts_r1:.1f}, {best_msls_r1:.1f})"
    )
else:
    if args.resume_author:
        util.load_model_weights_only(model, args.resume_author, device=args.device, strict=True)
        logging.info(
            f"Loaded author/full-model weights from {args.resume_author}; "
            "optimizer, epoch, and early-stopping state were not resumed."
        )
    best_pitts_r1 = -1.0
    best_msls_r1 = -1.0
    start_epoch_num = not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Getting GSVCities
train_dataset = get_GSVCities(
    base_path=args.training_dataset,
    cities='all',
    image_size=tuple(args.resize),
    synthetic_ratio=args.synthetic_ratio,
    training_subsets=args.training_subsets,
    tmp_group=args.tmp_group,
)
logging.info(
    f"Training set: root={args.training_dataset}, subsets={args.training_subsets}, "
    f"tmp_group={args.tmp_group}, "
    f"cities={len(train_dataset.cities)}, places={len(train_dataset)}, "
    f"images={train_dataset.total_nb_images}"
)

train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False}

#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.7, last_epoch=-1)
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses=[]
    for images, place_id in tqdm(ds):       
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)

        descriptors = model(images.to(args.device))
        descriptors = descriptors.cuda()
        loss = loss_function(descriptors, labels) # Call the loss_function we defined above
        del descriptors

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        del loss
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")

    eval_names_this_epoch = intermediate_eval_dataset_names
    logging.info("Running intermediate evaluation on datasets: %s", eval_names_this_epoch)

    epoch_recalls = {}
    for dataset_name in eval_names_this_epoch:
        eval_ds = eval_datasets[dataset_name]
        recalls, recalls_str = test.test(args, eval_ds, model, test_method=args.test_method)
        epoch_recalls[dataset_name] = recalls
        logging.info(f"Recalls on {dataset_name} {eval_ds}: {recalls_str}")

    pitts_r1 = float(epoch_recalls["pitts30k"][0])
    msls_r1 = float(epoch_recalls["Msls_740"][0])
    selection_tuple = (pitts_r1, msls_r1)
    best_selection_tuple = (best_pitts_r1, best_msls_r1)
    is_best = selection_tuple > best_selection_tuple
    next_best_pitts_r1 = pitts_r1 if is_best else best_pitts_r1
    next_best_msls_r1 = msls_r1 if is_best else best_msls_r1
    next_not_improved_num = 0 if is_best else not_improved_num + 1

    state = {
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_recalls": epoch_recalls,
        "selection_score": selection_tuple,
        "best_pitts_r1": next_best_pitts_r1,
        "best_msls_r1": next_best_msls_r1,
        "pitts_r1": pitts_r1,
        "msls_r1": msls_r1,
        "not_improved_num": next_not_improved_num,
    }

    torch.save(state, join(args.save_dir, "last_model.pth"))

    if is_best:
        torch.save(state, join(args.save_dir, "best_model.pth"))
        logging.info(
            "Saved best checkpoint by (Pitts30k R@1, MSLS R@1): "
            f"({best_pitts_r1:.1f}, {best_msls_r1:.1f}) -> ({pitts_r1:.1f}, {msls_r1:.1f})"
        )
        best_pitts_r1 = next_best_pitts_r1
        best_msls_r1 = next_best_msls_r1
        not_improved_num = next_not_improved_num
    else:
        not_improved_num = next_not_improved_num
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: "
            f"best (Pitts30k, MSLS) = ({best_pitts_r1:.1f}, {best_msls_r1:.1f}), "
            f"current = ({pitts_r1:.1f}, {msls_r1:.1f})"
        )
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(
    f"Best checkpoint tuple: (Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f})"
)
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

if final_eval_dataset_names:
    logging.info(
        "Testing best_model.pth on final datasets (excluding SF-XL): %s",
        final_eval_dataset_names,
    )
    best_model_state_dict = torch.load(
        join(args.save_dir, "best_model.pth"),
        map_location=args.device,
        weights_only=False,
    )["model_state_dict"]
    model.load_state_dict(best_model_state_dict)
    model.eval()
    for dataset_name in final_eval_dataset_names:
        eval_ds = eval_datasets[dataset_name]
        recalls, recalls_str = test.test(args, eval_ds, model, test_method=args.test_method)
        logging.info(f"Final recalls on {dataset_name} {eval_ds}: {recalls_str}")
