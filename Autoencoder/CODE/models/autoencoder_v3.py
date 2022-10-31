import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEnc(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        #enc6, dec1 부분에서 256이었던 것을 10으로 수정.
        self.enc6 = nn.Sequential(
            nn.Conv2d(128, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(16, 128, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(True),
        )

        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 1, padding=0),
        )

        # latent space output 을 인풋으로 받아 적당히 mlp layer 수정
        self.mlp_head = nn.Sequential(
            nn.Linear(256, 36),
            nn.ReLU(True),
            nn.Linear(36, 36),
            nn.ReLU(True),
            nn.Linear(36, 36),
            nn.ReLU(True),
            nn.Linear(36, 16),
            nn.ReLU(True),
            nn.Linear(16, num_labels)
        )

        self.encoder = nn.ModuleList(
            [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6]
        )

        self.decoder = nn.ModuleList(
            [self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.dec6]
        )

        self.CELoss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss(reduction='mean')
        self.CorrLoss = self.correlation_loss
        self.CosSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6)


    def correlation_loss(self, gt_speckle, pred_speckle):
        b, c, h, w = gt_speckle.size()
        corr = torch.einsum("b c i j, b c j k  -> b c i k", gt_speckle, pred_speckle)
        corr = F.normalize(corr.flatten(2), dim=-1)
        corr_loss = torch.mean(corr)
        return -corr_loss


    def encoding(self, x):
        feats = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            feats.append(x)
        return feats


    def decoding(self, x):
        feats = []
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            feats.append(x)
        return feats


    def forward(self, x, label):
        enc_feats = self.encoding(x)
        feat = enc_feats[-1]
        dec_feats = self.decoding(feat)
        speckle = dec_feats[-1]

        prob = self.mlp_head(feat.flatten(1))
        ce_loss = self.CELoss(prob, label)
        l1_loss = self.L1Loss(x, speckle)
        loss = ce_loss + l1_loss

        return prob, loss, ce_loss, l1_loss
