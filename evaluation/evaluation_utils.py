# define auxiliary functions --> NOTE: TO CHECK REDUNDANCY WITH OTHER UTILS SCRIPTS
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
from math import hypot
import numpy as np
import pandas as pd
import torch
import cv2

#from keras import backend as K
#import tensorflow as tf

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def model_inference(data_loader, model, vae_flag = False):
    model.eval()
    preds = []
    targets = []
    images = []
    if vae_flag:
        for ix, (x, y) in enumerate(data_loader):
            print("batch {} on {}".format(ix, len(data_loader)))
            with torch.no_grad():
                mu, sigma, segm, (mu_p, sigma_p) = model(x.to(device))
                #recon_loss, au, ne, au_1ch, ne_1ch = loss_VAE_rec(mu_p.to(device), sigma_p.to(device), x.to(device))
                preds.extend(segm.detach().cpu())
                images.extend(x)
                targets.extend(y)
                torch.cuda.empty_cache()

    else:
        for ix, (x, y) in enumerate(data_loader):
            with torch.no_grad():
                print("batch {} on {}".format(ix, len(data_loader)))
                results = model(x.to(device)).cpu().detach()
                preds.extend(results)
                images.extend(x)
                targets.extend(y)
                torch.cuda.empty_cache()
    return images, targets, preds


def post_processing(preds, targets, th=0.3, min_obj_size=2,
                    foot=4, area_threshold=6, max_dist=3):
    '''
    preds: array of tensor (ch, h, w)
    targets: array of tensor (ch, h, w)

    return:
    processed_preds: array of tensor (ch, h, w)
    targets: array of tensor (ch, h, w)
    '''
    if len(preds[0].shape) > 2:
        ix = np.argmin(preds[0].shape)
        if ix != 0:
            raise Exception("channels are not on the first dimension \
                            or are more than the spatial dimension")
        preds_t = [(np.squeeze(x[0:1, :, :]) > th) for x in preds]

    if len(targets[0].shape) > 2:
        ix = np.argmin(targets[0].shape)
        if ix != 0:
            raise Exception("channels are not on the first dimension \
                            or are more than the spatial dimension")
        targets = [np.squeeze(x[0:1, :, :]) for x in targets]

    processed_preds = []
    for p, t in zip(preds_t, targets):
        labels_pred, nlabels_pred = ndimage.label(p)
        processed = remove_small_holes(labels_pred, area_threshold=area_threshold, connectivity=1,
                                       in_place=False)
        processed = remove_small_objects(processed, min_size=min_obj_size,
                                         connectivity=1, in_place=False)
        labels_bool = processed.astype(bool)
        distance = ndimage.distance_transform_edt(processed)

        maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
        local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones((foot, foot)),
                                    exclude_border=False,
                                    labels=np.squeeze(labels_bool))
        local_maxi = remove_small_objects(
            local_maxi, min_size=min_obj_size, connectivity=1, in_place=False)
        markers = ndimage.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=np.squeeze(labels_bool),
                           compactness=1, watershed_line=True)
        processed_preds.append(labels.astype("uint8")*255)

    return processed_preds, targets

def compute_metrics(targets, preds_processed, filenames):

    metrics = pd.DataFrame(columns=["TP", "FP", "FN", "Target_count", "Predicted_count"])
    for ix, (p, t) in enumerate(zip(preds_processed, targets)):
        # extract predicted objects and counts
        pred_label, pred_count = ndimage.label(p)
        pred_objs = ndimage.find_objects(pred_label)

        # compute centers of predicted objects
        pred_centers = []
        for ob in pred_objs:
            pred_centers.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                                 (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

        # extract target objects and counts
        targ_label, targ_count = ndimage.label(t)
        targ_objs = ndimage.find_objects(targ_label)

        # compute centers of target objects
        targ_center = []
        for ob in targ_objs:
            targ_center.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                                (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

        # associate matching objects, true positives
        tp = 0
        fp = 0
        for pred_idx, pred_obj in enumerate(pred_objs):

            min_dist = 30  # 1.5-cells distance is the maximum accepted
            TP_flag = 0

            for targ_idx, targ_obj in enumerate(targ_objs):

                dist = hypot(pred_centers[pred_idx][0]-targ_center[targ_idx][0],
                             pred_centers[pred_idx][1]-targ_center[targ_idx][1])

                if dist < min_dist:

                    TP_flag = 1
                    min_dist = dist
                    index = targ_idx

            if TP_flag == 1:
                tp += 1
                TP_flag = 0

                targ_center.pop(index)
                targ_objs.pop(index)

        # derive false negatives and false positives
        fn = targ_count - tp
        fp = pred_count - tp

        # update metrics dataframe
        metrics.loc[filenames[ix]] = [tp, fp, fn, targ_count, pred_count]

    F1_score, MAE, MedAE, MPE, accuracy, precision, recall = F1Score(metrics)
    summary = pd.DataFrame(columns=["F1_score", "MAE", "MedAE", "MPE", "accuracy", "precision", "recall"])
    summary.loc['summary'] = [F1_score, MAE, MedAE, MPE, accuracy, precision, recall]
    return(metrics, summary)

def compute_metrics_global(targets, preds_processed):
    report_list = []
    tp = 0
    fp = 0
    fn = 0
    total_pred_counts = 0
    total_true_count = 0
    for p, t in zip(preds_processed, targets):
        pred_label, pred_rgb = ndimage.label(p)
        total_pred_counts += pred_rgb
        pred_objs = ndimage.find_objects(pred_label)
        # extract target objects and counts
        true_label, true_count = ndimage.label(t)
        total_true_count += true_count
        true_objs = ndimage.find_objects(true_label)
        # compute centers of predicted objects
        pred_centers = []
        for ob in pred_objs:
            pred_centers.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                                 (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))
        # compute centers of target objects
        targ_center = []
        for ob in true_objs:
            targ_center.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                                (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))
        # associate matching objects, true positives
        tp_objs = []
        for pred_idx, pred_obj in enumerate(pred_objs):
            min_dist = 30  # 1.5-cells distance is the maximum accepted
            TP_flag = 0
            for targ_idx, targ_obj in enumerate(true_objs):
                dist = hypot(pred_centers[pred_idx][0] - targ_center[targ_idx][0],
                             pred_centers[pred_idx][1] - targ_center[targ_idx][1])
                if dist < min_dist:
                    TP_flag = 1
                    min_dist = dist
                    index_targ = targ_idx
                    index_pred = pred_idx
            if TP_flag == 1:
                tp += 1
                TP_flag = 0
                tp_objs.append(pred_objs[index_pred])
                targ_center.pop(index_targ)
                true_objs.pop(index_targ)
        # derive false negatives and false positives
        for pred_obj in pred_objs:
            if pred_obj not in tp_objs:
                fp += 1
        for targ_obj in true_objs:
            fn += 1
        print("tp {} fn {} fp {}".format(tp, fn, fp))

    return tp, fn, fp, total_true_count, total_pred_counts


def F1Score(metrics):
    # compute performance measure for the current quantile filter
    tot_tp_test = metrics["TP"].sum()
    tot_fp_test = metrics["FP"].sum()
    tot_fn_test = metrics["FN"].sum()
    tot_abs_diff = abs(metrics["Target_count"] - metrics["Predicted_count"])
    tot_perc_diff = (metrics["Predicted_count"] -
                     metrics["Target_count"])/(metrics["Target_count"]+10**(-6))
    accuracy = (tot_tp_test + 0.001)/(tot_tp_test +
                                      tot_fp_test + tot_fn_test + 0.001)
    precision = (tot_tp_test + 0.001)/(tot_tp_test + tot_fp_test + 0.001)
    recall = (tot_tp_test + 0.001)/(tot_tp_test + tot_fn_test + 0.001)
    F1_score = 2*precision*recall/(precision + recall)
    MAE = tot_abs_diff.mean()
    MedAE = tot_abs_diff.median()
    MPE = tot_perc_diff.mean()

    return(F1_score, MAE, MedAE, MPE, accuracy, precision, recall)
    
### Plotting utils
def plot_thresh_opt(df, model_name, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    line = df.plot(y="F1", linewidth=2, markersize=6, legend=False),
    line = plt.title('$F_1$ score: threshold optimization', size =18, weight='bold')
    line = plt.ylabel('$F_1$ score', size=15)
    line = plt.xlabel('Threshold', size=15 )
    line = plt.axvline(df.F1.idxmax(), color='firebrick', linestyle='--')
    if save_path:
        outname = save_path / 'f1_score_thresh_opt_{}.png'.format(model_name[:-3])
        _ = plt.savefig(outname, dpi = 900, bbox_inches='tight' )
    return line