import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.metrics import ConfusionMatrixDisplay

def color_print(text:str, color:str='green', bold:bool=False):
    print(colored(text, color, attrs=['bold'] if bold else None))

def read_json(file_path:str):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_gt_polygon(gt_annotations, idx:int):
    # GT segmentation mask
    gt_seg = gt_annotations["segmentation"]
    gt_mask = gt_seg[idx] # e.g., [[787, 317, 851, 315, 839, 366, 776, 372]]
    gt_mask = gt_mask[0] # remove the outer list

    # split the list into x and y coordinates (COCO format: x1, y1, x2, y2, ..., xn, yn)
    x = gt_mask[::2]
    y = gt_mask[1::2]

    # add the first point to the last point to close the polygon
    x.append(x[0])
    y.append(y[0])

    # convert x and y to [[x1,y1], [x2,y2], ..., [xn,yn]] format
    gt_mask = np.array(list(zip(x, y)))
    gt_mask = Polygon(gt_mask)
    return gt_mask, (x,y)

def get_pred_polygon(pred_masks, image_idx:int, idx:int):
    # image file name of the predicted mask
    image_name = pred_masks[image_idx][1]

    # predicted mask
    pred_mask = pred_masks[image_idx][0][idx]

    # convert tensor to numpy array
    pred_mask = pred_mask.numpy()

    # find edge (exterior) points of predicted mask
    edge = np.zeros_like(pred_mask)
    edge[1:,:] = edge[1:,:] | (pred_mask[1:,:] != pred_mask[:-1,:])
    edge[:-1,:] = edge[:-1,:] | (pred_mask[1:,:] != pred_mask[:-1,:])
    edge[:,1:] = edge[:,1:] | (pred_mask[:,1:] != pred_mask[:,:-1])
    edge[:,:-1] = edge[:,:-1] | (pred_mask[:,1:] != pred_mask[:,:-1])
    edge = edge * pred_mask

    # extract x and y coordinates of the edge points
    y,x = np.where(edge == 1)
    edge = np.array(list(zip(x, y)))
    pred_mask = Polygon(edge).buffer(1)
    return image_name, pred_mask, edge

def compute_iou(gt_annotations, pred_masks):
    # compute IoUs
    # reference: https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
    ious = []
    tps_temp, fps_temp, fns_temp = [], [], []
    tps, fps, fns = [], [], []
    # for each image
    for i in tqdm(range(len(pred_masks)), desc="Computing IoU..."):
        # for each mask in the image
        for j in range(pred_masks[i][0].shape[0]): 
            image_name_pred, pred_mask, pred_edge = get_pred_polygon(pred_masks, i, j)
            # compute IoU with all GT masks in the same image
            for k in range(len(gt_annotations)): 
                # if the image names match
                if gt_annotations["file_name"][k] == image_name_pred: 
                    gt_mask, (gt_x,gt_y) = get_gt_polygon(gt_annotations, k)
                    intersection = gt_mask.intersection(pred_mask)
                    tp = intersection.area
                    fp = pred_mask.area - tp
                    fn = gt_mask.area - tp
                    iou = tp / (tp + fp + fn) # iou = intersection.area / union.area
                    ious.append(iou)
                    tps_temp.append(tp); fps_temp.append(fp); fns_temp.append(fn)
            # locate the index of the highest IoU
            idx = np.argmax(ious)
            tps.append(tps_temp[idx]) # TP of the maks of the highest IoU
            fps.append(fps_temp[idx]) # FP of the maks of the highest IoU
            fns.append(fns_temp[idx]) # FN of the maks of the highest IoU
            # reset temporary lists
            ious = []
            tps_temp, fps_temp, fns_temp = [], [], []
    tps, fps, fns = np.array(tps), np.array(fps), np.array(fns)
    return tps, fps, fns

def confusion_matrix_by_area(tps, fps, fns):
    # compute confusion matrix by area
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = np.sum(tps) # TP
    confusion_matrix[0, 1] = np.sum(fns) # FN
    confusion_matrix[1, 0] = np.sum(fps) # FP
    confusion_matrix[1, 1] = 0
    confusion_matrix = confusion_matrix.astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Positive", "Negative"])
    disp.plot(values_format="d", cmap="viridis")
    plt.title(f"Confusion Matrix by Pixel Area (Number of Pixels)")
    plt.show()
    return confusion_matrix

def evaluation_metrics_by_area(confusion_matrix):
    # compute recall, precision, and F-1 score
    recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1]) # TP / (TP + FN)
    precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0]) # TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    acc = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
    color_print(f"Recall={recall*100:.2f}%, Precision={precision*100:.2f}%, F-1 Score={f1*100:.2f}%, Accuracy={acc*100:.2f}%", color='green', bold=True)
    return recall, precision, f1, acc
    