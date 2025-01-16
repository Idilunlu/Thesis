# this class is a modified version of the original macenko method
# macenko method is a stain normalization method that normalizes the stain of the image
# it is used to normalize the stain of the image to make it easier to analyze

import torch
from torchstain.base.normalizers.he_normalizer import HENormalizer
from torchstain.torch.utils import cov, percentile

class TorchMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io, beta):
        I = I.permute(1, 2, 0)
        OD = -torch.log((I.reshape((-1, I.shape[-1])).float() + 1) / Io)
        ODhat = OD[torch.mean(OD, dim=1) > beta, :]
        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)
        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))
        return HE

    def __find_concentration(self, OD, HE):
        Y = OD.T
        X = torch.linalg.lstsq(HE, Y).solution[:HE.size(1)]
        return X

    def __compute_matrices(self, I, Io, alpha, beta, HE=None, upper=True):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)
        if HE is None:
            _, eigvecs = torch.linalg.eigh(cov(ODhat.T), UPLO='U' if upper else 'L')
            eigvecs = eigvecs[:, [1, 2]]
            HE = self.__find_HE(ODhat, eigvecs, alpha)
        C = self.__find_concentration(OD, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        if isinstance(I, list):
            I_list = I
            HEs = []
            maxCs = []
            for I in I_list:
                HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)
                HEs.append(HE)
                maxCs.append(maxC)
            HE = torch.stack(HEs, dim=0)
            maxC = torch.stack(maxCs, dim=0)
            HE_mean = torch.mean(HE, dim=0)
            maxC_mean = torch.mean(maxC, dim=0)
            self.HERef = HE_mean
            self.maxCRef = maxC_mean
        else:
            HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)
            self.HERef = HE
            self.maxCRef = maxC

    def fit_source(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)
        return HE, maxC

    def normalize(self, I, Io=240, Io_out=240, alpha=1, beta=0.15, HE=None, stains=True):
        c, h, w = I.shape
        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta, HE)
        C *= (self.maxCRef / maxC).unsqueeze(-1)
        Inorm = Io_out * torch.exp(-torch.matmul(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.T.reshape(h, w, c).int()
        H, E = None, None
        if stains:
            H = torch.mul(Io_out, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()
            E = torch.mul(Io_out, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()
        return Inorm, H, E
