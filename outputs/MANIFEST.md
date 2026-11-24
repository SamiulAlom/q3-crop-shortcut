| file | notebook | cell | description |
|---|---|---|---|
| raw_data_manifest.csv | 01_audit_masks_split_CPU | download | Datasets: URL, pinned commit, remote head at run time, fetch time |
| pv_exclusions_by_class.csv | 01_audit_masks_split_CPU | exclusions | PlantVillage, 30 classes: images kept and excluded by reason and class |
| pv_mask_threshold_sensitivity.csv | 01_audit_masks_split_CPU | exclusions | Mask area rule at thresholds 8, 16 (used) and 32: images below 5% and above 95%, median share, mean absolute change against 16 |
| pv_mask_profile.csv | 01_audit_masks_split_CPU | exclusions | Quantiles of mask area share (threshold 16, and 8 and 32 for sensitivity), share at 224 px, and alignment of the two copies inside the mask |
| mask_overlay_sheet_1.jpg | 01_audit_masks_split_CPU | exclusions | 50 random usable images: colour image with the mask edge in red (left) and its filled background (right) |
| mask_overlay_sheet_2.jpg | 01_audit_masks_split_CPU | exclusions | 50 random usable images: colour image with the mask edge in red (left) and its filled background (right) |
| mask_overlay_sheet_3.jpg | 01_audit_masks_split_CPU | exclusions | 50 random usable images: colour image with the mask edge in red (left) and its filled background (right) |
| mask_overlay_sheet_4.jpg | 01_audit_masks_split_CPU | exclusions | 50 random usable images: colour image with the mask edge in red (left) and its filled background (right) |
| mask_overlay_list.csv | 01_audit_masks_split_CPU | exclusions | The 200 images shown on the overlay sheets, in reading order |
| mask_overlay_lowest_alignment.jpg | 01_audit_masks_split_CPU | exclusions | The 50 kept images with the lowest alignment correlation: colour image with mask edge (left), filled background (right) |
| mask_overlay_lowest_alignment_list.csv | 01_audit_masks_split_CPU | exclusions | Images on mask_overlay_lowest_alignment.jpg, in reading order |
| mask_overlay_excluded.jpg | 01_audit_masks_split_CPU | exclusions | Up to 50 images excluded for alignment or mask area |
| mask_overlay_excluded_list.csv | 01_audit_masks_split_CPU | exclusions | Images on mask_overlay_excluded.jpg, in reading order |
| near_duplicate_profile.csv | 01_audit_masks_split_CPU | near duplicates | pHash threshold profile: groups, largest group, candidate pairs by kind, share of same-leaf pairs and count of random different-leaf pairs within each distance (-1 = identity links only) |
| phash_distance_histogram.csv | 01_audit_masks_split_CPU | near duplicates | pHash distance of all same-leaf pairs and of random pairs from different known leaves |
| near_duplicate_pairs_d0-0.jpg | 01_audit_masks_split_CPU | near duplicates | Up to 30 random candidate pairs at pHash distance 0-0 that do not share a known leaf identifier |
| near_duplicate_pairs_d1-2.jpg | 01_audit_masks_split_CPU | near duplicates | Up to 30 random candidate pairs at pHash distance 1-2 that do not share a known leaf identifier |
| near_duplicate_pairs_d3-4.jpg | 01_audit_masks_split_CPU | near duplicates | Up to 30 random candidate pairs at pHash distance 3-4 that do not share a known leaf identifier |
| near_duplicate_pairs_d5-6.jpg | 01_audit_masks_split_CPU | near duplicates | Up to 30 random candidate pairs at pHash distance 5-6 that do not share a known leaf identifier |
| near_duplicate_pairs_d7-8.jpg | 01_audit_masks_split_CPU | near duplicates | Up to 30 random candidate pairs at pHash distance 7-8 that do not share a known leaf identifier |
| near_duplicate_pair_sheets.csv | 01_audit_masks_split_CPU | near duplicates | The pairs shown on the near-duplicate sheets, in reading order |
| pv_split.csv | 01_audit_masks_split_CPU | split | Frozen split: one row per usable image of the 30 classes (cache row idx, class, file, leaf id, merged group, partition; train_only = the group holds an image without a leaf identifier or two classes, training only) |
| attribution_subset.csv | 01_audit_masks_split_CPU | split | Fixed attribution subset: up to 30 test images per class, one per merged leaf group |
| pv_split_by_class.csv | 01_audit_masks_split_CPU | split | Images and leaf groups per class and partition |
| near_duplicate_candidates_by_partition.csv | 01_audit_masks_split_CPU | split | Unmerged near-duplicate candidate pairs (pHash distance above the grouping threshold, up to 8) by partition pair and distance |
| pv_inventory.csv.gz | 01_audit_masks_split_CPU | inventory | PlantVillage, 30 classes: one row per colour image (row order = cache row), twin, leaf id, mask and alignment values, hashes, exclusion reason |
| pv_class_table.csv | 01_audit_masks_split_CPU | inventory | PlantVillage, 38 classes: files, images with a leaf identifier, distinct leaves, and whether the class is used (deviation 1) |
| plantdoc_inventory.csv | 01_audit_masks_split_CPU | plantdoc | PlantDoc: one row per image file (folders, git blob, size, SHA-256, decoded size, read error, mapped PlantVillage class, byte-identical duplicates) |
| plantdoc_class_mapping.csv | 01_audit_masks_split_CPU | plantdoc | PlantDoc folder to PlantVillage class mapping (empty = not used), with image counts |
| audit_summary.json | 01_audit_masks_split_CPU | summary | Run summary of notebook 01: parameters, versions, counts, exclusion counts, split and subset hashes |
| val_donors.csv | 02_train_GPU | learning rate | Fixed donor of every validation image (cache rows), used for own-regime validation images in training and reused by notebook 03 |
| lr_choice.csv | 02_train_GPU | learning rate | Learning-rate choice per architecture: ORIG regime, seed 11, best validation loss at each rate, chosen rate, measured training time |
| train_log.csv | 02_train_GPU | assemble | Per-epoch training loss, validation loss, accuracy and macro F1 (validation images built by the run's own regime), learning rate and time, all 36 runs |
| train_runs.csv | 02_train_GPU | assemble | One row per run: regime, seed, learning rate, best epoch (lowest validation loss), validation scores on own-regime and on original images, reload check, training time, peak GPU memory, checkpoint hash |
| pretrained_weights.csv | 02_train_GPU | assemble | Pretrained weight files: Hugging Face repository, pinned commit, SHA-256, normalisation constants |
| train_environment.json | 02_train_GPU | assemble | Training environment, run signature (configuration, input, weight, code and package fingerprints), chosen learning rates, memory probe and total time |
| check_subset.csv | 03_attribution_GPU | checks | Validation images used for the construction checks and their five fixed donors (cache rows); whether the texture-only image could be built |
| construction_checks.csv | 03_attribution_GPU | checks | Construction checks per checkpoint on validation images: accuracy on originals, background, leaf and artefact effects (probability and logit margin), tail shares, descriptive outline-only, texture-only and leaf-removed accuracy, pass flags |
| construction_checks_by_class.csv | 03_attribution_GPU | checks | Background and leaf effects and accuracy by class, per checkpoint (validation check subset) |
| model_eligibility.csv | 03_attribution_GPU | checks | Eligibility per architecture and seed pair, fixed before any test image is scored (plan section 4) |
| checks_environment.json | 03_attribution_GPU | checks | Environment and fixed rules of the construction-check stage |
| attribution_scores.csv.gz | 03_attribution_GPU | assemble | Off-leaf share S per checkpoint, image of the attribution subset, method and target class (predicted or true); native-grid S for Grad-CAM; map status (0 used, 1 zero mass, 2 non-finite); background area share; original images and own-regime images (sensitivity iii) |
| image_effects.csv.gz | 03_attribution_GPU | assemble | Per checkpoint and image of the attribution subset: predicted class and its probability, background and leaf effects over five fixed swaps (probability and logit margin), Integrated Gradients completeness values |
| classification_test.csv | 03_attribution_GPU | assemble | Test-partition accuracy and macro F1 per checkpoint on original, background-swapped, leaf-swapped and leaf-removed images; for swaps also accuracy against the donor's label |
| deletion_insertion.csv.gz | 03_attribution_GPU | assemble | Deletion and insertion areas under the curve, ORIG checkpoints, per image, map (methods and model-independent controls) and fill |
| zero_mass_maps.csv | 03_attribution_GPU | assemble | Number of maps with zero mass and with non-finite values (both replaced by the uniform map) per checkpoint, method, target and image kind |
| deletion_insertion_curves.csv | 03_attribution_GPU | assemble | Mean deletion and insertion curves (probability of the predicted class at each of 29 steps) per ORIG checkpoint, map and fill |
| test_predictions.csv.gz | 03_attribution_GPU | assemble | Predicted class index per checkpoint and test image on original, background-swapped, leaf-swapped and leaf-removed images (class order = sorted class names) |
| test_donors.csv | 03_attribution_GPU | assemble | Fixed donor (cache row) of every test image for the swapped test images |
| ig_completeness.csv | 03_attribution_GPU | assemble | Integrated Gradients completeness: relative difference between the summed attribution and the logit difference to the black baseline, median and 95th percentile per checkpoint |
| example_maps.npz | 03_attribution_GPU | assemble | Maps (112 x 112, predicted class) of six fixed images of the attribution subset for the seed-11 checkpoints, for the example figure |
| attribution_donors.csv | 03_attribution_GPU | assemble | Fixed donors (cache rows) of the attribution-subset images for swaps and own-regime images |
| attribution_environment.json | 03_attribution_GPU | assemble | Environment, fixed settings, frozen input hashes and timing of the attribution stage |
| contrasts_primary.csv | 04_statistics_figures_CPU | primary | The six pre-specified contrasts: estimate (unweighted mean of class means), 95% percentile bootstrap interval, centred-bootstrap p-value, Holm-adjusted p-value over six slots |
| contrasts_sensitivity.csv | 04_statistics_figures_CPU | primary | Sensitivity analyses of the contrasts: true class explained, Grad-CAM on its native grid, own-regime images (estimates and intervals, no p-values) |
| detection_difference_by_model.csv | 04_statistics_figures_CPU | primary | Detection difference D per architecture (95% interval) and per seed pair, each method |
| detection_difference_all_pairs_exploratory.csv | 04_statistics_figures_CPU | exploratory | EXPLORATORY (plan deviation 5): D per architecture, seed pair and method for all pairs, eligible or not; 95% bootstrap interval over leaves for fixed checkpoints; no p-values; with each BG checkpoint's validation background effect, leaf effect and leaf-removed accuracy |
| off_leaf_share.csv | 04_statistics_figures_CPU | secondary | Mean off-leaf share S per architecture, method and regime (eligible seeds for BG and LEAF, all seeds for ORIG), 95% interval, excess over the area share, and for BG the below-uniform rate with its tie rate |
| secondary_a_dose_response.csv | 04_statistics_figures_CPU | secondary A | Secondary A: Spearman correlation over images between the excess off-leaf share and the background effect (probability and logit margin) on ORIG models, mean over seeds with 95% interval, per-seed values, spread and ties of the effect |
| secondary_b_deletion_insertion.csv | 04_statistics_figures_CPU | secondary B | Secondary B: deletion and insertion areas under the curve on ORIG models per architecture, map and fill (mean over seeds, 95% interval); for methods, the control-adjusted score (method minus leaf silhouette, method minus edge map). Lower deletion and higher insertion mean a better map |
| classification_summary.csv | 04_statistics_figures_CPU | tables | Test accuracy and macro F1 per architecture and regime on original, background-swapped, leaf-swapped and leaf-removed images (mean, minimum and maximum over seeds); accuracy against the donor's label for swaps |
| construction_checks_summary.csv | 04_statistics_figures_CPU | tables | Construction checks per architecture and regime, mean over seeds, with the number of checkpoints that passed (validation images) |
| exploratory_off_leaf_by_class.csv | 04_statistics_figures_CPU | exploratory | Exploratory: mean off-leaf share of ORIG models by class, method and family, with the class's mean background area share |
| exploratory_off_leaf_right_wrong.csv | 04_statistics_figures_CPU | exploratory | Exploratory: mean off-leaf share of ORIG models for right and wrong predictions (maps pooled over seeds and images) |
| fig2_off_leaf_share.pdf | 04_statistics_figures_CPU | figures | Figure 2: off-leaf share S by method, architecture and training regime (ORIG: mean over all seeds; BG and LEAF: mean over the eligible seed pairs, shown for eligible architectures only; 95% bootstrap interval over leaves); dashed line = uniform map |
| fig2_off_leaf_share.png | 04_statistics_figures_CPU | figures | Figure 2 as PNG (600 dpi) |
| fig2_data.csv | 04_statistics_figures_CPU | figures | Numbers plotted in Figure 2; seeds = checkpoints averaged (ORIG: all seeds; BG and LEAF: eligible seed pairs only) |
| statistics_environment.json | 04_statistics_figures_CPU | environment | Environment, settings, method-test result, eligible seed pairs and input hashes of the statistics notebook |
| fig3_example_maps.pdf | 05_example_figure_GPU | figures | Figure 3 (illustration, no test): maps of ViT-S/16 seed pair 22 (eligible) for the first three of the six pre-fixed example images, predicted class, original images; each map on its own colour scale; S from attribution_scores.csv |
| fig3_example_maps.png | 05_example_figure_GPU | figures | Figure 3 (illustration, no test): maps of ViT-S/16 seed pair 22 (eligible) for the first three of the six pre-fixed example images, predicted class, original images; each map on its own colour scale; S from attribution_scores.csv |
| fig3_data.csv | 05_example_figure_GPU | figures | Images, checkpoints, predicted classes and off-leaf shares shown in Figure 3 (stored and recomputed S) |
| fig3_environment.json | 05_example_figure_GPU | environment | Environment and fixed choices of the Figure 3 notebook |
