# Analysis plan: paper-03 (QPAIN 2027)

Draft 2, Taha, 2026-11-22 (after the council, `discussions/analysis-plan.md`; decision `decisions/001-analysis-plan-council.md`). Written before any data is downloaded or any model is trained. **Approved by the human on 2026-11-22 ("Approve plan", checkpoint 2)** and committed before any data is touched (integrity rule 4). The human did not answer the question on Noyan 2022, so the plan's default stands: the preprint is not cited and Molhi 2026 carries the claim (section 12). The human picked gap 1 with gap 2 as a secondary analysis on 2026-11-22 ("Gap 1 with 2").

## 1. Working title
Do the explanations hold? Testing saliency maps of vision transformers and CNNs against a known background shortcut in crop-disease images. (The human approves the final title.)

Target: QPAIN 2027, track "Explainable AI and Trustworthy Intelligent Systems" (4-6 pages, IEEE template, double-blind, deadline 20 December 2026). Type: method-evaluation study; no new model is proposed, so Toni is not used. Reporting: a reproducibility checklist (NeurIPS style). No registry is used: the paper says "pre-specified, committed before data access", never "registered".

## 2. Question
Crop-disease papers add saliency maps to show that the model "looks at the lesion". PlantVillage classifiers are known to use the image background (section 12). A saliency method that is any use should show this.

- **Primary question.** When a classifier is built so that its prediction depends on the background and not on the leaf, do saliency maps move off the leaf, compared with a classifier built to depend on the leaf and not on the background? Which methods (Grad-CAM, Integrated Gradients, attention rollout, transformer attribution) show the shortcut, on which architectures?
- **Secondary question A (dose-response).** For classifiers trained in the usual way, does the share of attribution that falls off the leaf track the measured effect of the background on each prediction?
- **Secondary question B (gap 2).** On deletion and insertion curves, do the methods beat maps that never see the model (leaf silhouette, edge map, centred blob)?

Hypotheses (two-sided, stated before any result; either direction, or no detectable difference, is a reportable result):
- H1 (one per method): the off-leaf attribution share on the background-reliant model differs from that on the leaf-reliant model, on the same images.
- H2: on the two transformers, the difference in H1 for transformer attribution differs from that for Grad-CAM.
- H3: for Grad-CAM, the difference in H1 on transformers differs from that on CNNs.

## 3. Data
Facts read on the repositories on 2026-11-22 (GitHub API and README files); each is verified again on the files at step 3 (`data_sources.md`). Nothing has been downloaded.

| Role | Dataset | What it is | Access and licence | Open points for step 3 |
|---|---|---|---|---|
| Main | PlantVillage (Mohanty 2016, doi:10.3389/fpls.2016.01419), `github.com/spMohanty/PlantVillage-Dataset`, folders `raw/color` and `raw/segmented` | 54,306 leaf images, 14 crops, 38 classes, photographed on a plain surface; a segmented copy with the background removed; a `leaf_id` linking photographs of the same physical leaf | open download; licence given as CC BY-SA 3.0 on the dataset card, no licence file in the repository | licence wording; whether every colour image has a segmented twin and a `leaf_id`; pixel alignment of the two copies; repository size about 2 GB |
| Exploratory only | PlantDoc (Singh 2020, CODS-COMAD), `github.com/pratikkayal/PlantDoc-Dataset` | about 2,600 field and web images, 28 class folders, no leaf masks | open download; CC BY 4.0 (licence file) | class mapping to PlantVillage; image count |

**Leaf mask.** The mask of a colour image is the set of non-black pixels of its segmented twin (threshold fixed in notebook 01 on a sample before any model; small holes filled, largest components kept, by one rule for all images). Notebook 01 audits it: pairing by file name, alignment (correlation of the two copies inside the mask), mask area share. Images are excluded from every analysis if there is no twin, if alignment fails, or if the mask covers less than 5% or more than 95% of the image; counts by class are reported. The masks are the dataset owners' automatic segmentation, not expert outlines; 200 random overlays are saved for a visual check, and the paper states the limitation.

**Split.** The unit is the physical leaf (`leaf_id`). Exact duplicates (SHA-256 of decoded pixels) are found first; duplicates that link two leaf groups join them, also across classes. Near duplicates are audited before the split with a perceptual hash: the distance threshold is profiled on pairs known to show the same leaf (same `leaf_id`) against random pairs from different leaves, and matched pairs are looked at before the threshold is trusted. Pairs from different groups under the threshold join their groups. Stop rule: if the threshold cannot separate the two kinds of pair, or the largest merged group would hold more than 1% of the images, no near-duplicate merging is done, the candidate counts by partition pair are published and the paper states the limitation. The threshold is chosen after looking at images and before any model; the paper says so. Groups are split 70% train, 15% validation, 15% test, stratified by class, seed 20261122. An image without a `leaf_id` is its own group and is used for training only: it is never in the validation checks, the test partition or the attribution subset. The count is reported. The split file (image, group, partition) is saved with its SHA-256 and never changed. The test partition is used once, for the final evaluation; all tuning uses validation data.

**Attribution subset.** 30 test images per class (all, if a class has fewer), one image per merged leaf group (a group that spans classes gives one image in total), drawn once with seed 20261122 before any model is trained: about 1,140 images. Every model and method is scored on the same images.

**Background model of an image.** The background of each image, with the leaf region filled by normalised convolution (Gaussian blur of the background pixels divided by the blurred background mask, sigma fixed at 1/8 of the image side), computed once and cached. This uses no network.

Three image constructions, all built from the cached pieces with no network:
- **Background swap:** the leaf of image x on the filled background of a donor image.
- **Leaf swap:** the leaf of a donor image on the filled background of x.
- **Leaf removal:** the filled background of x alone.
Donors are drawn uniformly from the same partition as x, whatever their class (same-class donors allowed), so that the donor's content says nothing about the recipient's label. A leaf swap whose donor has the recipient's class is still labelled by the background. Default compositing: fill sigma 1/8 of the image side, hard mask edge. A faint trace of x's leaf outline can remain in its filled background; section 9 says what that does to the claims.

Pre-processing, the same everywhere: resize to 224 x 224, ImageNet normalisation.

## 4. Models
Four public ImageNet-pretrained networks from `timm`, weights pinned by repository commit and SHA-256: ResNet-50 and EfficientNet-B0 (CNNs), DeiT-Small/16 and ViT-Small/16 (transformers with global attention, so that attention rollout is defined). Three seeds each (11, 22, 33).

Three training regimes per architecture and seed, 38 classes each: 36 runs.
| Label | Training images | Label follows | Built to rely on |
|---|---|---|---|
| ORIG | original colour images | both | unknown mixture (what papers train) |
| LEAF | background swap, new random donor each epoch | the leaf | the leaf: the background carries no label information |
| BG | leaf swap, new random donor each epoch | the background | the background: the leaf carries no label information |

One protocol for all: AdamW, cosine schedule, 10 epochs, batch 64, mixed precision, cross-entropy, flips and mild colour jitter (applied after compositing). Learning rate chosen per architecture from {1e-4, 3e-4} by validation loss of the ORIG regime with seed 11, then fixed for every regime and seed. The checkpoint with the lowest validation loss (on validation images built by the model's own regime with fixed donors) is kept. Every cached output is bound to the hash of its checkpoint.

**Construction checks (on validation images, per seed, frozen before any test image is scored or any attribution computed).** The background effect of a model on an image is the mean, over 5 fixed background swaps, of the absolute change in its probability for the class it predicts on the original image; the leaf effect is the same with 5 leaf swaps. Absolute changes are used so that rises and falls do not cancel. Both effects are also computed on the logit margin (predicted-class logit minus the largest other logit) and reported, because probabilities of accurate models saturate. The **artefact effect** is the mean absolute change in that probability when only the compositing changes and the visible leaf and background content stay the same: the image rebuilt from its own leaf and its own filled background with (i) fill sigma 1/4 instead of 1/8 and (ii) a 3-pixel feathered edge instead of a hard edge, each against the default rebuild.
- A BG checkpoint passes if its accuracy on original images is at least 20% (chance is 2.6%), its mean background effect is at least 0.30, its mean leaf effect is at most 0.10 and its artefact effect is at most 0.10.
- A LEAF checkpoint passes if its accuracy on original images is at least 90%, its mean leaf effect is at least 0.30, its mean background effect is at most 0.10 and its artefact effect is at most 0.10.
- A seed pair (BG and LEAF of one architecture and seed) is eligible if both pass. An architecture is eligible if at least two of its three seed pairs are; its outcomes use the eligible pairs only. The eligibility table is written to `outputs/` and committed before notebook 03 scores any test image. A failed architecture is reported as failed and left out of H1-H3. If fewer than one CNN and one transformer are eligible, the primary question cannot be answered by this design; the paper reports that outcome, and secondary question A stays secondary. The thresholds are fixed here and are not changed after seeing results.
- A model can pass and still read the faint leaf outline left in a filled background; the artefact check bounds this only partly (section 9).

## 5. Attribution methods
All are computed for the model's predicted class on the original (uncomposited) image, in float32, as non-negative maps normalised to sum to one at 224 x 224 (coarse maps are upsampled bilinearly). A map with zero total mass is replaced by the uniform map and counted; the count is reported per method and model.
| Label | Method | Models |
|---|---|---|
| GC | Grad-CAM; target layer = the last layer whose output has a spatial size larger than 1 x 1 (last convolutional stage for CNNs; the tokens after the last block's first normalisation for transformers), fixed by this rule and checked for every model | all four |
| IG | Integrated Gradients, baseline = black in RGB space (pixel value 0 before normalisation), 32 steps, absolute value summed over channels | all four |
| ROLL | attention rollout (head-averaged attention with identity added, multiplied through the blocks; class-token row). It does not depend on the class explained; the paper calls it class-agnostic | transformers |
| TA | transformer attribution of Chefer et al. 2021 (gradient-weighted attention rollout, positive part) | transformers |

Model-independent control maps, the same for every model:
| Label | Map |
|---|---|
| SIL | the leaf mask (uniform inside, zero outside) |
| EDGE | Sobel gradient magnitude of the grey image |
| CENTRE | a centred Gaussian, sigma 1/4 of the image side |
| UNIF | uniform |

## 6. Outcomes
- **Off-leaf share S** of a map on an image: the sum of the map outside the leaf mask. A uniform map gives the background's area share a; the **excess off-leaf share** is S minus a. A leaf-reliant model with a faithful method should give S near 0; a background-reliant one, S near 1.
- **Primary: detection difference D** of a method on an architecture: mean over the attribution subset of S on the BG model minus S on the LEAF model, on the same images, averaged over seeds (seed k of BG is paired with seed k of LEAF). Every control map gives D = 0 exactly, because it does not depend on the model.
- **Secondary, reported beside D:** mean S for BG, LEAF and ORIG; the **below-uniform rate**, the share of images on which the map of the BG model has S below the background area share a (less off-leaf attribution than a blank map, while the model depends on the background), with the share of ties (S equal to a) reported separately.
- **Sensitivity analyses of D:** (i) explaining the true class instead of the predicted class for GC, IG and TA, so that BG and LEAF maps are for the same class; (ii) for Grad-CAM, S computed on the map's native grid with the fractional overlap of each cell with the mask, without upsampling; (iii) explaining each model on images built by its own regime (fixed donors) instead of the original image.
- **Secondary A:** on ORIG models, Spearman correlation over images between the excess off-leaf share and the background effect of section 4 (5 fixed swaps; probability and logit-margin versions), per method and architecture. The spread of the background effect and its share of ties are reported; a correlation with a constant input is reported as undefined.
- **Secondary B:** deletion and insertion areas under the curve on ORIG models (most relevant first, 28 steps of 1/28 of the pixels, fill = Gaussian blur; mean-colour fill as a sensitivity analysis), for the four methods and the four control maps. Control-adjusted score: method minus SIL, and method minus EDGE.
- **Classification:** accuracy and macro-F1 on the test partition for every model, on original, background-swapped, leaf-swapped and leaf-removed images.

## 7. Statistical analysis
- Level of summary: outcomes are averaged over the three seeds within an architecture; pooled values average the eligible architectures. Per-architecture and per-seed values are always printed next to pooled values.
- **Six pre-specified contrasts**, Holm-corrected together at 0.05:
  - H1-GC, H1-IG: D of Grad-CAM and of Integrated Gradients, pooled over the eligible architectures, against zero.
  - H1-ROLL, H1-TA: D of attention rollout and of transformer attribution, pooled over the eligible transformers, against zero.
  - H2: D of TA minus D of GC, pooled over the eligible transformers.
  - H3: D of GC on transformers minus D of GC on CNNs.
- Eligibility: H1-ROLL, H1-TA and H2 need at least one eligible transformer; H3 needs at least one eligible CNN and one eligible transformer. A contrast that cannot be estimated is reported as such and keeps its place in the correction: the Holm family always counts six.
- A method "detects the shortcut" in the paper's wording only if its H1 contrast is positive after correction **and** its below-uniform rate is reported next to it; a small positive D with a high below-uniform rate is described as such.
- Class weighting: every outcome is the unweighted mean of the 38 class means, so a class with fewer than 30 images counts as much as the others.
- Inference: bootstrap with 10,000 resamples of the images of the attribution subset, drawn within each class (one image per merged leaf group, so the image is the independent unit), the same resample for every model and method (paired). Percentile intervals. Two-sided p-value from the centred bootstrap. Seeds, architectures and trained checkpoints are fixed, not resampled: inference is conditional on these trained networks and is about new leaves, not about all networks. All random seeds are fixed and saved.
- Secondary outcomes get estimates and bootstrap intervals, no p-values.
- A method test is kept beside the statistics notebook and rerun after every edit: the bootstrap code against an explicit computation, and its false-positive rate under generated data with no effect. If the false-positive rate at 0.05 is above 0.075, p-values are not reported, only estimates and intervals, and this is written as a deviation before results.

## 8. Exploratory analyses (labelled as such in the paper)
- **Field images.** Accuracy of ORIG, LEAF and BG models on the PlantDoc classes that map to PlantVillage classes (mapping table fixed in notebook 01 before any model is trained). It asks whether removing the background's label information helps outside the laboratory. No masks exist for PlantDoc, so no off-leaf share is computed. First to be cut.
- Off-leaf share by class, and for wrong against right predictions of ORIG models.

Cut order if time is short: PlantDoc first, then secondary B's mean-colour fill, then EfficientNet-B0 (leaving one CNN and two transformers), then the third seed of the LEAF and BG regimes. One full training run and one full attribution run are timed before the rest is launched; a cut is made only from those measured times and is written in section 10 before any result.

## 9. What will not be claimed
- No claim that any method is "faithful" in general: the test covers one kind of shortcut (image background) in one laboratory dataset, four networks and four methods.
- No claim that a map on the leaf means the model uses the lesion: the mask separates leaf from background, not lesion from healthy tissue. Claims are about reliance on the off-leaf background against the leaf region, never about lesions.
- If a BG model reads the outline trace near the mask edge, attribution along the contour falls partly inside the mask; the off-leaf share then understates how well a method follows that model. The paper says this next to the BG results and shows example maps.
- The idea of testing attribution methods against models with a known or planted dependence is not new (section 12). The paper claims only its application: a naturally occurring background shortcut in crop-disease images, CNNs against transformers, with transformer-specific methods.
- The BG and LEAF models are constructions, not models anyone would deploy; the paper says so and reports the construction checks. Filled backgrounds can keep a faint outline of the removed leaf, and composites have hard mask edges; a model could use either. The construction checks bound this (a BG model with a large leaf effect fails), but do not remove it.
- No claim about field performance beyond the exploratory PlantDoc table.
- No claim of equivalence from a non-significant contrast.
- Paper-01 of this group (retinal OCT, edge-map control on deletion curves) is a different question; no text, figure or result is shared, and secondary B is kept secondary so that the two papers do not have the same design.

## 10. Deviations
**Deviation 1 (2026-11-22, before any image was downloaded and before any model; decision `decisions/002-leaf-id-coverage.md`, made by Taha acting as Sam while the human was away).** The repository's file list and leaf map show that 13,194 of the 54,305 colour images have no leaf identifier, and that eight classes have none at all (the four Corn classes, Grape healthy, Squash powdery mildew, Tomato target spot, Tomato mosaic virus). Changes to sections 3 and 4:
- The study uses the **30 classes that have leaf identifiers** (46,418 colour images before exclusions); the networks have 30 outputs. Chance accuracy is 3.3%, not 2.6%; the construction-check thresholds are unchanged.
- Images of those 30 classes without a leaf identifier (5,307) are training-only, as section 3 already says.
- The split is **60% train, 15% validation, 25% test** by leaf group (was 70/15/15), same seed, so that classes with few leaves have enough test leaves. The attribution subset stays "30 test images per class, one per leaf group, all if fewer"; expected size about 785 images (was about 1,140), at least about 9 per class.
- PlantDoc classes that map to a dropped class are not used.
- Twins are paired by exact file name, and by the identifier after the prefix where the colour file has no prefix; an image whose twin is ambiguous is excluded and counted.
Reason: only for these classes can the split guarantee that evaluation images come from leaves unseen in training. Effect on claims: results cover 30 of 38 classes and no maize; the paper says so. Wherever sections 6 and 7 say "38 class means", read 30.

**Deviation 2 (2026-11-22, before any image was opened; from Codex's review of notebook 01, `results/001-review-notebook-01.md`, decision `decisions/003-notebook-01-review.md`).**
- *Artefact check (section 4).* As written, the check rebuilt each image from its own leaf and its own filled background. With a hard edge that rebuild is the original image whatever the fill, so part (i) would always give zero. Replaced by: for each validation image and one fixed donor, the model's own-regime swap image is built three ways, (a) default (fill sigma 1/8, hard edge), (b) fill sigma 1/4, (c) 3-pixel feathered edge; the artefact effect is the larger of the mean absolute changes in the predicted-class probability, (b) against (a) and (c) against (a). For a BG model the swap is the donor's leaf on the image's filled background; for a LEAF model, the image's leaf on the donor's filled background. Thresholds unchanged (at most 0.10).
- *Groups that can be evaluated (section 3).* A merged group goes to validation or test only if every image in it has a leaf identifier and all are of one class. Any other group is used for training whole. This is stricter than section 3 and makes "an image without an identifier is never evaluated" hold row by row.
- *Mask threshold (section 3).* Fixed before any image was opened, not on a sample: a pixel is leaf if the largest channel of the segmented copy is above 16 (of 255); all enclosed holes are filled; components smaller than 5% of the largest are dropped. The audit reports the area rule at thresholds 8 and 32. The area rule (5% to 95%) must hold at the native size and at 224 x 224.
- *Alignment (section 3).* An image is excluded as misaligned if the correlation of grey values of the two copies inside the eroded mask is below 0.90, or if moving the colour image by 4 pixels in any of four directions gives a smaller mean absolute RGB difference than no move.
- *Near-duplicate threshold.* The manager sees pair sheets that include images later assigned to test; this is a look at images, not at any model output, and is said in the paper. The split and the attribution subset are written once; a later run must reproduce them byte for byte.

**Deviation 3 (2026-11-22, after the first audit run of notebook 01 and before any split or model; decision `decisions/005-alignment-rule.md`).** The alignment rule of deviation 2 compared pixel values of the two copies. The first run (`logs/local-run-01`) showed that the owners' segmented copy is registered with the colour image but has different colours: no pixel was identical inside the mask in the pairs inspected, the mean absolute difference had a median of 18 grey levels, and the rule excluded 28,467 of 46,418 images although the mask outlines on the overlay sheets follow the leaves. The rule is replaced, on evidence that involves no model:
- An image is registered if the correlation of the fine texture of the two copies (grey values minus their Gaussian blur, sigma 2 px, inside the mask eroded by 6 px) is at least 0.15 **and** higher than after moving the colour image by 4 px in each of four directions. Otherwise it is excluded as misaligned.
- Evidence (`decisions/evidence/005_alignment_rule.py`, 1,500 seeded images): true twins 0.228 to 0.92 and always above every shifted value; the same images moved by 4 px: none passes; the segmented copy of another image of the same class: at most 0.05. The threshold 0.15 lies between the two groups and was chosen from this sample before any split.
- What stays true: masks are the owners' automatic segmentation. The overlay sheets show cast shadows of the leaf on the surface outside the mask, and some masks that miss dark or glossy parts of a leaf or include a strip of shadow. Shadows are off-leaf pixels in every analysis: a BG model may read the leaf's shadow, and a saliency map on the shadow counts as off-leaf. The paper says so (section 9), and the outline-only and texture-only checks show how much a BG model depends on shape.

**Deviation 4 (2026-11-22, before any model; decision `decisions/007-notebook-04-review.md`).** The exploratory field-image analysis of section 8 (accuracy on PlantDoc) is cut, as the first item of the plan's cut order. Reasons: the paper has 4 to 6 pages; float32 scoring roughly doubles the attribution time; with deviation 1 six PlantDoc folders have no counterpart. Notebook 01 still publishes the PlantDoc inventory and class mapping. No PlantDoc image is scored by any model, and the paper makes no claim about field images.

**Clarifications fixed before training (2026-11-22; details the plan left open, none changes an outcome definition).**
- Inputs are normalised with each network's own pretrained mean and standard deviation (the ViT-Small weights use 0.5 and 0.5, the others the ImageNet values), not one set for all. "Black" for Integrated Gradients stays pixel value 0 before normalisation.
- ViT-Small/16 uses the `augreg_in21k_ft_in1k` weights (ImageNet-21k pretraining, ImageNet-1k fine-tuning), timm's default for that name; the other three are ImageNet-1k weights.
- Weight decay 0.05; colour jitter = brightness, contrast and saturation factors from 1 ± 0.10, each acting on every pixel by itself (contrast about mid-grey, not about the image mean, so pasted and kept pixels of a composite cannot influence each other); a donor is drawn uniformly from the same pool and may be the image itself; validation donors are fixed once with a seed.
- Disk: the cache of images, masks and filled backgrounds needs about 16 GB, not 6 GB.
- Construction checks (from Codex's review of notebook 03, decision `decisions/006-notebook-03-review.md`, before any training): accuracy on original images uses all validation images. The background, leaf and artefact effects use a class-balanced subset of at most 50 validation images per class, drawn once with a seed; each image has 5 fixed donors drawn from the whole validation partition, never the image itself; every effect is the unweighted mean of the class means, as in section 7. The artefact check uses the first of the five donors.
- Scoring precision: training uses mixed precision (bfloat16); every forward and backward pass used for the construction checks, the maps, the effects and the curves is float32 without TF32.
- Transformer attribution is the attention-and-gradient form (positive part of attention times its gradient, averaged over heads, accumulated through the blocks), which the paper names exactly.
- Deletion and insertion: pixels ordered by decreasing map value, exact ties by a fixed random ranking of the image's pixels; blur fill = Gaussian, sigma 11 px; the probability tracked is that of the class predicted on the original image.
- A map with zero mass, or with a non-finite value, is replaced by the uniform map and counted (the two cases separately), also on Grad-CAM's native grid.
- Below-uniform rate: S counts as tied with the area share a when they differ by at most 1e-9.
- Swaps on test images (classification table, per-image effects, own-regime images) draw their fixed donors from the test partition, never the image itself.

**Addition to the construction checks (2026-11-22, before any training; from Codex's review of notebook 02, `results/002-review-notebook-02.md`, decision `decisions/004-notebook-02-review.md`).** The four pass rules of section 4 are unchanged and stay the only gates. Passing them does not show that a BG model uses only the background, because the filled background keeps the outline of the removed leaf. Three descriptive checks are therefore added on validation images, reported for every checkpoint and never used to pass or fail a model:
- *Outline only:* accuracy on an image that is mid-grey everywhere with the leaf region of the image in a darker grey (no colour, no texture). High accuracy of a BG model here means it reads the outline.
- *Background texture only:* accuracy on an image made by tiling the 56 x 56 corner patch of the image that has the least leaf in it (images whose best corner holds any leaf pixel are left out and counted). High accuracy of a BG model here means it reads background colour and texture without any outline.
- *Leaf removed:* accuracy on the filled background alone.
The background, leaf and artefact effects are also reported by class and as the share of images with an effect below 0.05, not only as means. Claims stay limited to the interventions tested (section 9).

Platform (recorded here because it differs from the engine's default): all notebooks run on this PC (RTX 3060, 12 GB) under the human's standing order of 2026-11-19 for T4-class jobs; the methods section names this hardware. No approved-access data is involved.

### Deviation 5 (2026-11-22, written after the construction checks were frozen and before any test image was scored or any attribution output existed)
**What happened.** Construction checks, frozen in `notebooks/p3_frozen.json` (commit 341be4d): all 12 LEAF checkpoints pass. Ten of the 12 BG checkpoints fail one check only, "leaf effect at most 0.10" (observed 0.094 to 0.178; background effect 0.83 to 0.91). Eligible by the rule of section 4: ViT-S/16, seed pairs 22 and 33. No CNN is eligible.

**What does not change.** The thresholds, the eligibility rule and the six-contrast Holm family. H1-GC, H1-IG, H1-ROLL, H1-TA and H2 are estimated on ViT-S/16 with its two eligible seed pairs, and both seed estimates are printed. H3 cannot be estimated and is reported so; it keeps its place in the correction. The paper states that the question "CNN against transformer" was not answered by this design.

**What is added (exploratory, labelled so everywhere).** One table, "all pairs": D on original images for every architecture, seed pair and applicable method, whether eligible or not, class-balanced as in section 7, with a 95% paired class-stratified bootstrap interval (images resampled, checkpoints fixed) as pointwise uncertainty. No p-values, no pooled value over architectures or families, no comparison between families, outside the Holm family. Each row carries the pair's eligibility and the BG checkpoint's validation background effect, leaf effect and leaf-removed accuracy. The paper says with the table that it was added after the validation checks were known and before attribution outcomes were examined, that D in a non-eligible pair mixes the behaviour of the attribution method with an incompletely planted shortcut, and that the table cannot compare CNNs with transformers (the failed CNN checkpoints have larger leaf effects than the failed transformer checkpoints).

**Scope of inference, to be stated in the paper.** Intervals and p-values describe variation over sampled leaves for fixed checkpoints. They do not cover retraining (ViT-S/16 seed 11 failed the same check at 0.112) or other architectures.

**Wording.** The higher accuracy of BG checkpoints on leaf-removed images is reported as the net effect of that intervention only; it does not show what caused the gain and does not rule out leaf-derived cues in the filled background.

Council: `discussions/eligibility-outcome.md` (Codex and Gemini: option B, approve with changes; all changes accepted). Decision note: `decisions/010-eligibility-outcome.md`. Options not taken: raising the threshold (forbidden by section 4); retraining BG checkpoints under a changed construction (a new design chosen after seeing the checks).

### Deviation 6 (2026-11-23, written after the results existed; a reporting change, no analysis changed)
Section 6 says the paper "shows example maps" next to the BG results, for the case of a BG model reading the leaf outline. The six-page limit holds two figures (design, off-leaf share) and three tables (checks, contrasts, exploratory all pairs), so the paper has no example-map figure. The outline caveat stays in the text (limitation 3: it would lower D, so it cannot produce the positive result). The maps of the six fixed images for the seed-11 checkpoints are kept in `outputs/example_maps.npz` and are released with the code. No example was looked at to choose what to report. Decision note: `decisions/012-no-example-figure.md`.

**Deviation 6 revised (2026-11-23, the human's order "Add figure", written before any map was looked at).** The example-map figure is restored as Fig. 3, as illustration only: no test, no new estimate. Rules fixed now: ViT-S/16, seed pair 22 (the lower-numbered eligible pair); the first three, in order of idx, of the six example images drawn before training; predicted class on original images. The maps saved by notebook 03 are for seed 11, a pair that is not eligible, so the maps of these images are recomputed with the same functions in a new notebook `05_example_figure_GPU` (Codex review first), which stops unless every recomputed off-leaf share equals the stored one within 0.001. The figure prints the stored S of each map. Space is made by cutting text, not results. Decision note: `decisions/014-example-figure-restored.md`.

## 11. Outputs and reproducibility
Notebooks: `01_audit_masks_split_CPU` (download, pairing, masks, filled backgrounds, split, attribution subset, PlantDoc mapping), `02_train_GPU`, `03_attribution_GPU` (construction checks first, then maps and curves, cached), `04_statistics_figures_CPU`. Each writes to `outputs/` with a row in `outputs/MANIFEST.md`; every number in the manuscript comes from those files. Dataset commit, file hashes, the split hash, seeds, weights and package versions are frozen in the manifest. Codex reviews every notebook before it runs; each GPU notebook runs a 1% smoke test first. Attribution code is written fresh for this paper's file layout but follows the reviewed paper-01 functions where they apply (Grad-CAM, Integrated Gradients, deletion and insertion). Planned figures: Fig. 1 workflow (Jack), Fig. 2 off-leaf share by method, architecture and regime, Fig. 3 example maps for the three regimes; Table I classification and construction checks, Table II the six contrasts. Code is released with the paper.

## 12. Prior work this plan depends on
Read at abstract level only (`gaps.md`, 2026-11-22); a full-text prior-work check precedes drafting.
- PlantVillage background shortcut: Noyan 2022 (arXiv:2206.04374, preprint; search summary only); Molhi 2026 (doi:10.1109/cspa68262.2026.11517900): EfficientNet-B0, 98.97% on originals, 96.62% leaf-only, 23.58% background-only. Neither uses explanations.
- Explanation evaluation on single crops, without a known ground truth or a control map: Prabhu 2026 (doi:10.1080/21642583.2026.2734417), Kumar 2026 (doi:10.14445/23488549/ijece-v13i10p103), Kumar K. 2026 (doi:10.3389/fpls.2026.1965848).
- Mask-based explanation energy under a synthetic background change, clove grading: Nyalala 2026 (doi:10.4218/etrij.2026-0225). The closest design; it reports the energy ratio as a score and does not train models with a known dependence.
- Attribution methods tested on planted features outside plants: Zhou 2022 (AAAI, doi:10.1609/aaai.v36i9.21196; title read, content from memory, to be read before drafting).
- Named by Gemini in the council **from memory, not verified by anyone yet**: Adebayo et al. 2020 (NeurIPS, debugging tests for model explanations with spurious background correlations), Xiao et al. 2020/2021 (ICLR, ImageNet-9 backgrounds challenge with foreground-only and background-only training), Hesse et al. 2023 (CVPR, FunnyBirds). Each is looked up (existence, identifier, what it did) before drafting. The full texts of Molhi 2026, Prabhu 2026, Nyalala 2026 and Zhou 2022 are read before drafting to confirm that none reports an off-leaf attribution contrast between background-reliant and leaf-reliant models on PlantVillage.
- The PlantVillage shortcut is cited to Molhi 2026 (peer reviewed). Preprints are not cited in the manuscript (standing order) unless the human allows Noyan 2022 as the original source; the novelty wording is checked against all of them again in the week of submission.
