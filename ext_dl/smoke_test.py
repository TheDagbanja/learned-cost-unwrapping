"""Small forward/loss/backward checks for newly added baselines and WLS."""

from __future__ import annotations

import numpy as np
import torch

from src import utils
from .dmp_punet import DMPPUNetBaseline, dmp_punet_loss
from .modl_pu import MoDLPUBaseline, modl_pu_loss
from .unwrap_net import UnwrapNetBaseline, unwrap_net_loss
from .waveunet import WaveUNetBaseline, waveunet_loss
from .wls import unwrap_wls


def batch(size=64):
    psi = (torch.rand(1,size,size)*2-1)*torch.pi
    gt = psi + 2*torch.pi*torch.randint(-2,3,(1,size,size)).float()
    return {"wrapped":psi.unsqueeze(1), "absolute":gt.unsqueeze(1),
            "input":utils.encode_input(psi,include_grad=True), "coherence":torch.full((1,1,size,size),.8)}


def main():
    b=batch()
    cases=[(MoDLPUBaseline(4,widths=(8,16,32,64)),lambda o:modl_pu_loss(o,b,stage=2)),
           (UnwrapNetBaseline(4,widths=(8,16,32,64,64)),lambda o:unwrap_net_loss(o,b)),
           (DMPPUNetBaseline(4,widths=(8,16,32,64)),lambda o:dmp_punet_loss(o,b)),
           (WaveUNetBaseline(4,base=4),lambda o:waveunet_loss(o,b))]
    for model,loss_fn in cases:
        out=model(b); terms=loss_fn(out); terms["total"].backward()
        print(f"{model.__class__.__name__:24s} output={tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    psi=b["wrapped"][0,0].numpy(); pred=unwrap_wls(psi,np.full_like(psi,.8),maxiter=500)
    print(f"WLS output={pred.shape} finite={np.isfinite(pred).all()}")


if __name__=="__main__": main()
