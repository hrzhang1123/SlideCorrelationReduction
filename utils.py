import torch

## input is the list of slide indices
def gen_M_matrix(input, device='cuda'):
    tsize = len(input)
    M = torch.zeros(tsize,tsize).float().to(device)
    for ii in range(tsize-1):
        for jj in range(ii+1, tsize):
            if input[ii] == input[jj]:
                M[ii,jj] = 1
    return M


def criterion_CorrelationReduce(pred, gt, feat, ce_cri, M, c_w=1):
    ce_loss = ce_cri(pred, gt).mean()
    bs, chn = feat.shape

    ## other options of normalization are feasible
    norm_feat = torch.nn.functional.relu(torch.tanh(feat)) - 0.5

    corr_reduce_loss = (torch.mm(norm_feat, norm_feat.t()) * M / chn ).sum()
    all_loss = ce_loss + c_w * corr_reduce_loss
    return ce_loss, corr_reduce_loss, all_loss




