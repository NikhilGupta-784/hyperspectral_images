import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

def load_HSI(path):
    data = sio.loadmat(path)['indian_pines_corrected'].astype(np.float32)
    data=(data/9604)*255
    return data

def mask(img, mask_ratio = 0.5):
    mask = np.random.rand(*img.shape) > mask_ratio
    return img*mask

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return ((tensor - min_val) / (max_val - min_val + 1e-8))

class MaskedBandImageDataset(Dataset):
    def __init__(self, cube, masks_per_band=15):
        self.mask_img = []
        self.org_image = []
        h, w, b = cube.shape
        for band in range(b):
            band_img = cube[:, :, band]
            for _ in range(masks_per_band):
                masked = mask(band_img)
                self.mask_img.append(masked.astype(np.float32))
                self.org_image.append(band_img)

        self.transform = T.Compose([
            T.ToTensor(),
            # MinMaxNormalize()
        ])

    def __len__(self):
        return len(self.mask_img)

    def __getitem__(self, idx):
        masked_img = self.mask_img[idx]
        return self.transform(masked_img), self.transform(self.org_image[idx])
        # return masked_img, self.org_image[idx]# (input, target)


cube = load_HSI("Dataset/Indian_pines_corrected.mat")
dataset = MaskedBandImageDataset(cube, masks_per_band=15)
print("Dataset size:", len(dataset))  # Should be 3000

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    for data in dataloader:
        ip, tr = data
        print(f"{ip.shape=}    {tr.shape=}")
        print(f"{ip.min().item()=}    {tr.min().item()=}")
        print(f"{ip.max().item()=}    {tr.max().item()=}")




# x, _ = dataset[100]
# plt.imshow(x.squeeze(0), cmap='gray')
# plt.title("Masked Spectral Band Image")
# plt.show()

# ===========================

# band_index = 50
# original = cube[:, :, band_index]
# masked = mask(original, mask_ratio =0.4)
#
# # Plot side-by-side
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(original, cmap='gray')
# plt.title(f"Original Band {band_index}")
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(masked, cmap='gray')
# plt.title(f"Masked Band {band_index}")
# # plt.axis('off')
#
# plt.tight_layout()
# plt.show()
