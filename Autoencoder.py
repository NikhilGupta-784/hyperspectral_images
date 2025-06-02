import torch.nn as nn
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error as mae
from Autoencoder_dl import dataset, dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm
import piq


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, 3, stride = 2, padding=1), #145 - 73
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride= 2, padding=1), #73 - 37
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride= 2, padding= 1), #37 - 19
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride= 2, padding= 1, output_padding= 0), #19 - 37
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride= 2, padding= 1, output_padding= 0), #37 - 73
            nn.ReLU(),
            nn.ConvTranspose2d( 16, 1, 3, stride= 2, padding= 1, output_padding= 0), #73 - 145
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def evaluation(y_true, y_pred):
    y_true = y_true.squeeze().detach().cpu().numpy()
    y_pred = y_pred.squeeze().detach().cpu().numpy()

    psnr_val = psnr(y_true, y_pred, data_range= 1.0)
    ssim_val = ssim(y_true, y_pred, data_range= 1.0)
    mae_val = mae(y_true.flatten(), y_pred.flatten())

    return mae_val, psnr_val, ssim_val


class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        ssim_loss = 1 - piq.ssim(y_pred, y_true, data_range=1.0)  # SSIM ∈ [0,1]
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


def train(model, dataloader, device ='cuda', epochs = 20):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    criterion = SSIM_MSE_Loss(alpha=0.85)

    mae_ls, psnr_ls, ssim_ls = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        mae_e, psnr_e, ssim_e =0, 0, 0

        for x, y in tqdm(dataloader, desc=f'Epoch: {epoch+1}/{epochs}'):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            # y = torch.clamp(y, 0.0, 1.0)

            print("y_true min/max:", y.min().item(), y.max().item())
            print("y_pred min/max:", y_pred.min().item(), y_pred.max().item())

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            for i in range(x.size(0)):
                m,p,s = evaluation(y[i], y_pred[i])
                mae_e += m
                psnr_e +=p
                ssim_e += s


        n = len(dataloader.dataset)
        print(f"Loss: {total_loss: .4f} | MAE: {mae_e/n: .4f} | PSNR: {psnr_e/n:.2f} | SSIM: {ssim_e/n:.4f} ")
        mae_ls.append(mae_e/n)
        psnr_ls.append(psnr_e/n)
        ssim_ls.append(ssim_e/n)

    return mae_ls, psnr_ls, ssim_ls


if __name__ == "__main__":
    model = Autoencoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mae_list, psnr_list, ssim_list = train(model, dataloader, device=device, epochs=50)

    torch.save(model.state_dict(), "saved_model_AE\\AE_indian_pines4_ssimloss.pth")
    print("Model saved to AE_indian_pines.pth")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(mae_list, label='MAE')
    plt.plot(psnr_list, label='PSNR')
    plt.plot(ssim_list, label='SSIM')
    plt.title("Evaluation Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("autoencoder_metrics4_ssimloss.png")
    plt.show()
