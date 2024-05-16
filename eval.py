import os
import json
import argparse
import pandas as pd
import numpy as np
import torch

from utils import color_print, read_json, compute_iou, \
                    confusion_matrix_by_area, evaluation_metrics_by_area

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=100*200, help="Threshold for small object", required=False)
    args = parser.parse_args()

    # GT annotations
    fname = "ground-truth-coco/annotations.json"
    gt = read_json(fname)
    gt_images = pd.DataFrame(gt["images"])
    gt_annot = pd.DataFrame(gt["annotations"])

    # add images dataframe to annotations dataframe
    annotations = pd.merge(gt_annot, gt_images[["id", "height", "width", "file_name"]], left_on="image_id", right_on="id")
    annotations = annotations.drop(columns=["id_x", "id_y"]) # drop index columns
    annotations

    # Predictions
    pred_dir = "predictions" # parent directory of predictions
    image_names = os.listdir(pred_dir) # subdirectories (image file names) of predictions

    # full directory of outputs.json files
    outputs_dir = [os.path.join(pred_dir, image_names, "outputs.json") for image_names in image_names]
    color_print(f"Predicted Outputs directory:", color="yellow", bold=True)
    color_print(outputs_dir, color='yellow')

    # full directory of masks.npy files
    masks_dir = [os.path.join(pred_dir, image_names, "masks.npy") for image_names in image_names]
    color_print(f"Predicted Masks directory:", color="yellow", bold=True)
    color_print(masks_dir, color='yellow')

    # make a single dataframe for all outputs in the order of image file names
    outputs = pd.DataFrame()
    for i, output_dir in enumerate(outputs_dir):
        with open(output_dir, "r") as f:
            output = json.load(f)
        output = pd.DataFrame(output)
        output["file_name"] = image_names[i]
        outputs = pd.concat([outputs, output], axis=0)

    # make a single dataframe for all masks in the order of image file names, use torch tensor to save memory
    masks = []
    for i, mask_dir in enumerate(masks_dir):
        mask = torch.tensor(np.load(mask_dir))
        image_file_name = mask_dir.split("/")[1]
        mask = (mask, image_file_name) # tuple (mask, image file name)
        masks.append(mask)

    # compute IoUs, TPs, FPs, FNs
    ious, tps, fps, fns = compute_iou(annotations, masks, args.threshold)

    # compute confusion matrix
    confusion_matrix = confusion_matrix_by_area(tps, fps, fns, args.threshold)

    # compute evaluation metrics
    evaluation_metrics = evaluation_metrics_by_area(confusion_matrix)