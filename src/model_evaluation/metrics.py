import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
import os
import torch
import torchmetrics


def calculate_psnr(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return psnr(target, output, data_range=target.max() - target.min())


def calculate_ssim(output, target):
    output = output.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return ssim(target, output, data_range=target.max() - target.min(), channel_axis=-1)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_psnr = None
        self.best_ssim = None
        self.early_stop = False

    def __call__(self, epoch_loss, epoch_psnr, epoch_ssim):
        if self.best_loss is None:
            self.best_loss = epoch_loss
            self.best_psnr = epoch_psnr
            self.best_ssim = epoch_ssim
        elif (
            (epoch_loss < self.best_loss - self.min_delta)
            or (epoch_psnr > self.best_psnr + self.min_delta)
            or (epoch_ssim > self.best_ssim + self.min_delta)
        ):
            self.best_loss = epoch_loss
            self.best_psnr = epoch_psnr
            self.best_ssim = epoch_ssim
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def log_metrics_to_json(run_id, epoch, loss, psnr, ssim, total_time):
    # Create a dictionary for the current iteration's metrics
    log_entry = {
        "epoch": epoch,
        "total_time": total_time,
        "loss": loss,
        "psnr": psnr,
        "ssim": ssim,
    }

    a = []
    base_dir = "/home/linuxu/Documents/models/MaxSR-Tiny"
    metrics_dir = f"{base_dir}/{run_id}/metrics"

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    log_file = f"{metrics_dir}/metrics.json"
    if not os.path.isfile(log_file):
        a.append(log_entry)
        with open(log_file, mode="w+") as f:
            json.dump(a, f)
    else:
        with open(log_file, "r") as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(log_entry)
        with open(log_file, mode="w+") as f:
            json.dump(feeds, f)


def calculate_psnr_ssim_metrics(output, target, device):
    # Determine the data range based on the target image
    data_range = target.max() - target.min()

    # Initialize the metrics with the calculated data range
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(
        data_range=data_range
    ).cuda(device)

    # Calculate PSNR
    psnr_value = psnr_metric(output, target)

    # Calculate SSIM
    ssim_value = ssim_metric(output, target)

    return psnr_value.item(), ssim_value.item()
