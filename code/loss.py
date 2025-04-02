import torch
from torch.nn.functional import mse_loss, relu


def variance(z, gamma=1):
    return relu(gamma - z.std(0)).mean()    


def invariance(z1, z2):
    return mse_loss(z1, z2)


def covariance(z):
    n, d = z.shape
    mu = z.mean(0)
    cov = torch.einsum("ni,nj->ij", z-mu, z-mu) / (n - 1)
    off_diag = cov.pow(2).sum() - cov.pow(2).diag().sum()
    return off_diag / d


def vicreg_loss(rep1, rep2, la=25, mu=25, nu=1):
    var1, var2 = variance(rep1), variance(rep2)
    inv = invariance(rep1, rep2)
    cov1, cov2 = covariance(rep1), covariance(rep2)
    vicreg_loss = la*inv + mu*(var1 + var2) + nu*(cov1 + cov2)
    return vicreg_loss


def vicreg_loss(rep1, rep2, rep3, la=25, mu=25, nu=1):
    var1, var2, var3 = variance(rep1), variance(rep2), variance(rep3)
    inv1, inv2, inv3 = invariance(rep1, rep2), invariance(rep2, rep3), invariance(rep1, rep3)
    cov1, cov2, cov3 = covariance(rep1), covariance(rep2), covariance(rep3)
    vicreg_loss = la*(inv1 + inv2 + inv3) + mu*(var1 + var2 + var3) + nu*(cov1 + cov2 + cov3)
    return vicreg_loss


def vicreg_loss_without_inv(rep1, rep2, mu=25, nu=1):
    var1, var2 = variance(rep1), variance(rep2)
    cov1, cov2 = covariance(rep1), covariance(rep2)
    vicreg_loss = mu*(var1 + var2) + nu*(cov1 + cov2)
    return vicreg_loss


def vicreg_loss_without_inv(rep1, rep2, rep3, mu=25, nu=1):
    var1, var2, var3 = variance(rep1), variance(rep2), variance(rep3)
    cov1, cov2, cov3 = covariance(rep1), covariance(rep2), covariance(rep3)
    vicreg_loss = mu*(var1 + var2 + var3) + nu*(cov1 + cov2 + cov3)
    return vicreg_loss


def CE_loss(pred1, pred2, labels):
    CE = torch.nn.CrossEntropyLoss()
    CE_loss = CE(pred1, labels) + CE(pred2, labels)
    return CE_loss


def CE_loss(pred1, pred2, pred3, labels):
    CE = torch.nn.CrossEntropyLoss()
    CE_loss = CE(pred1, labels) + CE(pred2, labels) + CE(pred3, labels)
    return CE_loss