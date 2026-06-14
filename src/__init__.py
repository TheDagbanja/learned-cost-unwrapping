"""Physics-Guided Parameter-Efficient Adaptation for InSAR Phase Unwrapping.

Source package. See ``README.md`` and the research proposal for the full
method description. Submodules:

- ``utils``             : phase wrapping, edge gradients, ambiguity labels, seeding.
- ``dataset``           : InSAR-DLPU (.mat) and real GeoTIFF (Stream C) loaders.
- ``synthetic_engine``  : Okada/Mogi/atmosphere generator (Phase 2).
- ``models``            : Swin-T + LoRA backbone, U-Net decoder, unwrapper.
- ``integrator``        : differentiable weighted least-squares (Poisson) solver.
- ``losses``            : L_CE, L_cong, L_sparse, L_ent.
- ``train`` / ``evaluate`` : training and evaluation entry points.
"""

__version__ = "0.1.0"
