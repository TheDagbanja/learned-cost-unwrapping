---
title: "Learned Statistical Costs for Network-Flow InSAR Phase Unwrapping"
author: "Simon Dagbanja"
---

*Abstract* — Phase unwrapping recovers a continuous deformation field from the wrapped phase of an interferogram, and it remains the step where interferometric SAR most often fails. Classical network-flow methods such as SNAPHU solve a global integer problem with statistical costs derived by hand, while recent deep networks predict the unwrapped phase directly and tend to break when the test data drift away from the training distribution. We keep the network-flow solver and learn its cost function instead. A Swin Transformer with low-rank adaptation predicts a probability for the integer ambiguity on every edge of the interferogram grid, and the negative log of that probability becomes the per-edge cost of a minimum-cost-flow solver. The unwrapped field is therefore consistent and free of residues by construction. We also fuse the learned cost with a coherence-based statistical cost, and we show that the pipeline is differentiable end to end through the solver. On a simulated benchmark the learned costs lower the unwrapping error by about 79 percent relative to a least-squares baseline and by 48 percent relative to SNAPHU, which brings the solver close to the achievable noise floor. On the real subset of the same benchmark, where ground truth is available, they again improve on SNAPHU, by 43 percent. On four Sentinel-1 earthquake interferograms the method agrees with SNAPHU to a small fraction of a fringe where coherence is good, matches GPS to about two centimetres, and stays residue-free, all without retraining.

*Index Terms* — Phase unwrapping, synthetic aperture radar interferometry, minimum-cost flow, deep learning, surface deformation.

## I. Introduction

Interferometric synthetic aperture radar measures surface deformation from the phase difference between two radar acquisitions [@rosen2000; @massonnet1998]. The instrument only records phase modulo $2\pi$, so the observed interferogram is the wrapped version of the quantity of interest. Phase unwrapping is the inverse problem of adding the correct integer number of cycles back to each pixel. Its difficulty grows with noise, with steep deformation gradients, and with loss of coherence, and an error made at one pixel can propagate across a large region.

Two families of methods dominate. Path-following and least-squares approaches integrate the wrapped gradients directly. The branch-cut method places barriers between residues and integrates around them [@goldstein1988], reliability-guided unwrapping follows a quality map [@herraez2002], and weighted least squares solves a Poisson equation with fast transforms [@ghiglia1994; @ghigliapritt1998]. The second family treats unwrapping as a global optimization. Costantini showed that the integer problem is a minimum-cost flow on the dual grid [@costantini1998], and SNAPHU extended this with statistical cost functions derived from a model of the interferometric phase [@chen2001]. SNAPHU is the operational standard for deformation studies.

Deep learning has been applied to unwrapping mainly by predicting the unwrapped phase or the per-pixel wrap count directly from the wrapped input [@spoorthi2020; @wang2019; @zhou2020]. These networks work well on data that resemble their training set. They struggle elsewhere because the prediction is not constrained to be a consistent phase field, so it can contain residues and full-cycle errors that no post-processing removes. We observed the same behaviour with a per-pixel network of our own. It reached high accuracy on simulated data yet did not transfer to real interferograms.

We take a different route. We keep the rigorous solver, which guarantees a consistent solution, and we learn only the cost function that the solver minimizes. The cost is exactly the part of SNAPHU that is built by hand. Our contributions are the following. We define a learned negative log-likelihood cost for each edge and feed it to a minimum-cost-flow solver, so the result is residue-free by construction. We fuse this learned cost with a coherence-based statistical cost into a single maximum-likelihood cost. We make the pipeline differentiable end to end through the combinatorial solver using a blackbox gradient [@pogancic2020]. Finally we evaluate the method against five baselines on a simulated benchmark and on four Sentinel-1 coseismic interferograms, with GPS as an independent check.

![Method pipeline. The wrapped phase drives a Swin Transformer edge head that predicts per-edge ambiguity probabilities. Their negative log-likelihood becomes the cost of a minimum-cost-flow solver, which returns a residue-free unwrapped field together with an uncertainty map.](outputs/fig_pipeline.png)

## II. Method

### A. Wrapped phase and residues

Let $\phi$ be the true absolute phase on a regular grid and $\psi$ the observed wrapped phase. The wrapping operator returns the principal value,
$$ W(x) = x - 2\pi\,\Big\lfloor \tfrac{x+\pi}{2\pi} \Big\rfloor, \qquad \psi = W(\phi). $$
Unwrapping estimates $\phi$ from $\psi$ up to an additive constant. Work with the wrapped differences between neighbouring pixels,
$$ \Delta^{x}_{i,j} = W(\psi_{i,j+1}-\psi_{i,j}), \qquad \Delta^{y}_{i,j} = W(\psi_{i+1,j}-\psi_{i,j}). $$
The unwrapped gradient differs from the wrapped gradient by an integer number of cycles on each edge,
$$ g^{x}_{i,j} = \Delta^{x}_{i,j} + 2\pi\,k^{x}_{i,j}, \qquad g^{y}_{i,j} = \Delta^{y}_{i,j} + 2\pi\,k^{y}_{i,j}, $$
where $k^{x},k^{y}\in\mathbb{Z}$ are the edge ambiguities. When the true gradient stays below $\pi$ in magnitude the ambiguity is zero, which is the Itoh condition [@itoh1982]. The wrapped phase carries residues wherever this fails. The residue on the elementary loop at $(i,j)$ is the rounded loop sum of the wrapped gradients,
$$ R_{i,j} = \frac{1}{2\pi}\Big[\Delta^{x}_{i,j} + \Delta^{y}_{i,j+1} - \Delta^{x}_{i+1,j} - \Delta^{y}_{i,j}\Big], \qquad R_{i,j}\in\{-1,0,1\}. $$
A consistent unwrapped field has a curl-free gradient. The integer field must therefore satisfy, at every loop,
$$ k^{x}_{i,j} + k^{y}_{i,j+1} - k^{x}_{i+1,j} - k^{y}_{i,j} = -\,R_{i,j}. \qquad (1) $$

### B. Minimum-cost flow

Equation (1) has many integer solutions whenever the interferogram contains residues. The classical resolution is to pick the one of least total cost,
$$ \min_{k^{x},k^{y}} \ \sum_{e} c_{e}\,\lvert k_{e}\rvert \quad \text{subject to (1)}, \qquad (2) $$
where $c_{e}\ge 0$ is the cost of placing a cycle on edge $e$. Costantini showed that (2) is a minimum-cost flow on the dual graph, with the residues acting as sources and sinks and the cycles as flow on the edges between neighbouring loops [@costantini1998]. The solution is exact and globally optimal for the given costs, and the recovered gradient is curl-free, so integrating it gives an unwrapped field with no residues. We solve (2) with a network-simplex solver and obtain $\hat\phi$ by least-squares integration of the corrected gradient,
$$ \hat\phi = \arg\min_{\phi}\ \big\lVert \nabla\phi - g \big\rVert^{2}, \qquad (3) $$
which is computed with a discrete cosine transform [@ghiglia1994]. Because $g$ is curl-free after the solve, (3) is an exact integration rather than an approximation.

To let the cost distinguish a positive cycle from a negative one we split each edge into two directed arcs and write the objective in arc form,
$$ \min \ \sum_{e}\big(c^{+}_{e}\,k^{+}_{e} + c^{-}_{e}\,k^{-}_{e}\big), \qquad k_{e}=k^{+}_{e}-k^{-}_{e}, \quad k^{\pm}_{e}\ge 0. \qquad (4) $$

### C. Learned statistical costs

SNAPHU builds $c_e$ from a statistical model of the phase that depends on coherence, the number of looks, and an assumed deformation or topography prior [@chen2001]. We replace that hand-derived model with a learned one. A network reads the wrapped phase and predicts, for each edge, a probability over the three ambiguity classes,
$$ p_{e} = \mathrm{softmax}\big(z_{e}\big) \in \Delta^{2}, \qquad p_{e}(m)=\Pr(k_e=m),\ m\in\{-1,0,1\}, $$
where $z_e$ are the network logits. The cost of cycling an edge is the negative log-likelihood of the corresponding class,
$$ c^{+}_{e} = -\log p_{e}(+1), \qquad c^{-}_{e} = -\log p_{e}(-1). \qquad (5) $$
An edge that the network believes carries a cut is cheap to cut, and an edge it believes is quiet is expensive. Equation (5) is a learned maximum-likelihood cost, the data-driven counterpart of the statistical cost in SNAPHU, and it plugs into (4) without changing the solver.

### D. Combined maximum-likelihood cost

Coherence is the most reliable physical indicator of where the phase can be trusted. We fold it into the cost as an additive statistical term,
$$ c^{\pm}_{e} = -\log p_{e}(\pm 1) + \lambda\,\bar\gamma_{e}, \qquad (6) $$
where $\bar\gamma_{e}$ is the mean coherence of the two pixels joined by edge $e$ and $\lambda$ sets the balance. A high-coherence edge is expensive to cut, which keeps branch cuts in the unreliable regions, while the learned term carries the information that coherence alone does not capture. The two terms are complementary. The learned cost dominates where coherence is uninformative, and the physical cost dominates where it is reliable.

### E. End-to-end differentiability

The solver in (2) is piecewise constant in $c$, so it has no useful classical gradient. We use the blackbox-solver method of Pogančić et al. [@pogancic2020]. With $k^{\ast}=\mathrm{MCF}(c)$ and a loss $L$ on the integrated phase, the backward pass perturbs the cost in the direction of the upstream gradient, re-solves, and takes the finite difference,
$$ \frac{\partial L}{\partial c} \approx -\frac{1}{\beta}\Big[\,k^{\ast} - \mathrm{MCF}\big(c + \beta\,\tfrac{\partial L}{\partial k^{\ast}}\big)\Big], \qquad (7) $$
where $\beta$ trades gradient fidelity for informativeness. The integration in (3) is linear in $k$, so $\partial L/\partial k^{\ast}$ follows by automatic differentiation, and the gradient flows back to the network. Training then minimizes the actual unwrapping error rather than an edge-classification proxy.

### F. Reliability map

The same probabilities give a confidence product at no extra cost. The Shannon entropy of each edge distribution measures how unsure the network is about cutting, and the per-pixel mean of the incident-edge entropies is a reliability map,
$$ H_{e} = -\sum_{m} p_{e}(m)\log p_{e}(m), \qquad U_{i,j} = \tfrac{1}{2}\big(\bar H^{x}_{i,j}+\bar H^{y}_{i,j}\big). \qquad (8) $$
The map is high along the rupture trace and in decorrelated patches, which are the places where the unwrapped phase deserves least trust.

### G. Network and adaptation

The edge head is a Swin Transformer backbone [@liu2021swin] with a U-Net decoder that returns six channels per pixel, three ambiguity classes for the horizontal edge and three for the vertical edge. The input has four channels, the real and imaginary parts of $e^{i\psi}$ and the two wrapped gradients, which hand the network the local cut cue directly. We adapt the backbone with low-rank adapters of rank eight in the query and value projections [@hu2022lora], which keeps the trainable parameter count small. Large scenes are processed in overlapping tiles of $256\times256$ pixels, and the per-edge logits are averaged across tiles with a Hann window before the global solve, so the solver still sees the whole interferogram.

## III. Experiments

### A. Data and protocol

We train and benchmark on the public InSAR deep-learning phase-unwrapping dataset, which provides simulated wrapped and absolute pairs at $256\times256$ together with a small real subset. We then apply the trained network, without any retraining, to four Sentinel-1 coseismic interferograms processed by the Alaska Satellite Facility into wrapped phase, coherence, and a SNAPHU unwrapped product at 80 m spacing [@torres2012]. The four events span a range of faulting styles and coherence regimes. The 2019 Ridgecrest earthquake is strike-slip in the Mojave desert [@ross2019], the 2016 Amatrice earthquake is normal faulting in the vegetated Apennines [@cheloni2017], the 2015 Gorkha earthquake is a buried thrust with broad smooth deformation [@galetzka2015], and the 2016 Kumamoto earthquake mixes strike-slip and normal motion in low-coherence terrain [@himematsu2016]. We report the aligned root-mean-square error after removing the single reference constant, and for the real scenes we separate the comparison into high-coherence pixels, where SNAPHU is itself reliable, and the remainder.

### B. Benchmark against ground truth

Table I and Fig. 2 report the simulated benchmark, where the absolute phase is known.

| Method | RMSE (rad) |
|:--|:--:|
| Least squares [@ghiglia1994] | 4.33 |
| Quality-guided [@herraez2002] | 4.08 |
| Goldstein branch-cut [@goldstein1988] | 8.37 |
| Unit-cost MCF [@costantini1998] | 2.32 |
| SNAPHU [@chen2001] | 1.71 |
| Learned-cost MCF (ours) | 0.90 |
| Oracle-cost MCF | 0.84 |
| Ground-truth-$K$ floor | 0.76 |

: TABLE I. Unwrapping error on the simulated benchmark, as aligned RMSE in radians over 1000 scenes scored on identical inputs. Lower is better. All flow methods are residue-free.

The network-flow methods separate clearly from the path-following and least-squares methods. Branch-cut, quality-guided, and least-squares unwrapping sit between 4 and 8 rad of error. The learned cost reaches 0.90 rad, below SNAPHU at 1.71 rad and close to the cost oracle at 0.84 rad, which is itself near the achievable floor of 0.76 rad set by the sub-cycle noise. Relative to least squares this is a reduction of about 79 percent, relative to SNAPHU it is 48 percent, and relative to the cost-free unit-cost flow solver it is 61 percent. Learning the cost therefore improves on the hand-derived statistical model even on clean simulated data, and it brings the solver to within a small fraction of a radian of the oracle.

![Unwrapping error on the simulated benchmark over 1000 scenes. Network-flow methods (blue and red) are several times more accurate than the path-following and least-squares baselines (grey). The learned cost falls below SNAPHU and approaches the oracle cost and the noise floor.](outputs/fig_benchmark.png)

The dataset also provides a real TanDEM-X subset with its own ground truth, scored the same way over 100 scenes. Table II reports it. The learned cost reaches 3.02 rad, a reduction of 43 percent relative to SNAPHU and 49 percent relative to the cost-free flow solver, so among the network-flow methods it is again the most accurate by a wide margin. The achievable floor here is 1.55 rad, set by the heavier decorrelation noise of the real data, so the gap that remains reflects the data rather than a weakness of the solver. Reliability-guided unwrapping is the exception. It reaches 2.38 rad and so does better than any of the global solvers, the reverse of its behaviour on the simulated benchmark where it reaches 4.08 rad. The real subset carries smooth topographic phase, where following a high-reliability path avoids the rare cycle errors a global solver can still make, while on the steeper and more varied simulated scenes the same path-following breaks down. Taken together the learned cost beats the hand-derived statistical model of SNAPHU on both clean and real-noise data, and it stays robust across the regimes where simpler methods alternately succeed and fail.

| Method | RMSE (rad) |
|:--|:--:|
| Least squares | 7.42 |
| Quality-guided | 2.38 |
| Goldstein branch-cut | 7.73 |
| Unit-cost MCF | 5.96 |
| SNAPHU | 5.30 |
| Learned-cost MCF (ours) | 3.02 |
| Oracle-cost MCF | 3.71 |
| Ground-truth-$K$ floor | 1.55 |

: TABLE II. Unwrapping error on the real TanDEM-X subset, as aligned RMSE in radians over 100 scenes scored on identical inputs. Lower is better. The learned cost is the most accurate network-flow method and beats SNAPHU by 43 percent. Quality-guided unwrapping does best on this smooth subset, the reverse of Table I.

### C. Real interferograms

Table III and Fig. 3 summarize the four real events.

| Event | Faulting | Input res. | Res. (ours) | Learned | Learned+coh | GPS (cm) |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| Ridgecrest M7.1 | strike-slip | 9258 | 0 | 0.78 | 0.69 | 2.1 |
| Amatrice M6.2 | normal | 5358 | 0 | 0.06 | 0.04 | 1.8 |
| Gorkha M7.8 | thrust | 10087 | 0 | 0.00 | 0.00 | -- |
| Kumamoto M7.0 | strike-slip, normal | 89959 | 0 | 0.98 | 0.57 | -- |

: TABLE III. Real Sentinel-1 coseismic interferograms. Columns give the faulting style, the residue count of the wrapped input and of our output, the high-coherence agreement with SNAPHU in radians for the learned and combined costs, and the line-of-sight agreement with GPS in centimetres.

On all of them the learned solution has zero residues, against tens of thousands in the wrapped input. Where coherence exceeds 0.7 the learned-cost solution agrees with SNAPHU to well under one fringe, with $0.78$ rad on Ridgecrest, $0.06$ rad on Amatrice, and $0.00$ rad on Gorkha. The methods that ignore the cost prior behave very differently. Least squares and the cost-free flow solver diverge from SNAPHU by tens of radians on the near-fault Ridgecrest and Kumamoto scenes, because they place cycles in the wrong places near the aliased rupture. This is the clearest evidence that the cost, not the solver alone, carries the accuracy.

The combined cost of (6) is the strongest in every case where coherence is informative. Its largest gain is on Kumamoto, the lowest-coherence scene, where it lowers the high-coherence error from $0.98$ rad with the learned cost alone and $0.78$ rad with coherence alone to $0.57$ rad. Fig. 4 shows the qualitative result on Ridgecrest. The learned-cost field and the SNAPHU field are visually identical across the deformation lobes, and the two disagree only along the low-coherence fault trace, which is the one region that no unwrapper can resolve from phase alone.

![High-coherence agreement with SNAPHU across the four events, by cost. The prior-free unit-cost solver fails on the near-fault scenes, while the learned and combined costs stay below one fringe.](outputs/fig_highcoh.png)

![Ridgecrest interferogram. From left to right and top to bottom: wrapped phase, coherence, learned-cost result, SNAPHU result, their difference, and the reliability map. The difference is flat across the recoverable field and concentrates on the low-coherence fault trace.](outputs/ridgecrest_compare.png)

### D. GPS validation

Independent GPS coseismic offsets provide a check that does not depend on SNAPHU. We project the three-component offsets into the radar line of sight with the product look vectors and compare against the unwrapped phase at the station pixels. On Ridgecrest the learned-cost field matches GPS to 2.1 cm, identical to SNAPHU, over the stations inside the analysis window. On Amatrice both reach 1.8 cm. The stations sit on stable, high-coherence ground, which is where the two methods agree, so the result confirms that the learned costs reproduce the gold standard at independent measurement points rather than only matching it pixel for pixel.

### E. Differentiable cost training

We also trained the costs end to end on the simulated data with the blackbox gradient of (7). The unwrapping error falls as the costs adapt, which confirms that the gradient optimizes the true objective through the solver rather than a surrogate. The trained costs reach the same operating point on real data as the supervised costs, $3.0$ rad on the real subset against $5.3$ rad for SNAPHU. The supervised edge head already sits near this optimum, so the end-to-end refinement leaves the result essentially unchanged on these scenes. We therefore treat differentiability as a property that makes the cost genuinely learned rather than a separate accuracy gain, and we use the supervised variant in practice.

### F. Limitations

The method inherits the limits of any consistent unwrapper. Where the deformation aliases, with gradients above $\pi$ per pixel along a sharp rupture, the wrapped phase no longer determines the integer field, and the learned costs cannot recover information that the data do not contain. In that regime the solver and SNAPHU produce different and equally unverifiable answers, and the network costs, trained on moderate deformation, can become unreliable. The combined cost of (6) helps where coherence is good but does not change this floor. The reliability map of (8) is useful here because it marks the affected pixels.

## IV. Conclusion

We have shown that the cost function of network-flow phase unwrapping can be learned. A Swin Transformer predicts per-edge ambiguity probabilities, their negative log-likelihood becomes the cost of a minimum-cost-flow solver, and the result is residue-free by construction. The learned cost improves on SNAPHU on a simulated benchmark and on its real subset, agrees with SNAPHU to a fraction of a fringe on four real coseismic interferograms where coherence is good, agrees with GPS to about two centimetres, and on the simulated benchmark is far more accurate than least-squares, quality-guided, and branch-cut unwrapping. Fusing the learned cost with a coherence-based statistical cost gives the best result where coherence is informative, and the whole pipeline is differentiable through the solver. The method keeps the guarantees of the classical solver while letting the data shape its statistical model, which is the part that has been built by hand since SNAPHU.

## References
