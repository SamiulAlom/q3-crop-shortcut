# Data sources: paper-03 — checkpoint 3

Written 2026-11-22 by Taha, acting as Sam while the human is away. No image has been downloaded or opened. "Verified" means read on 2026-11-22 from the repository itself: the GitHub API, the repository's file list (names only) and its small text files. Evidence: `decisions/evidence/003_source_records.out.txt`, `004_file_list_audit.out.txt`, `004_per_class_loader_rule.csv`.

## 1. PlantVillage (main: training, validation, test)
- **Name:** PlantVillage Dataset, the copy released with Mohanty, Hughes and Salathé 2016.
- **Owner:** S. P. Mohanty, M. Salathé (Digital Epidemiology Lab, EPFL); images from the PlantVillage project (D. P. Hughes).
- **URL:** https://github.com/spMohanty/PlantVillage-Dataset (verified). **Version pinned:** commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, 5 February 2026, branch `master` (verified). The repository was changed in 2026 (a loader and a dataset card were added), so the commit is pinned and read again at the end of the download.
- **Licence:** `[CHECK]` The repository has no licence file and GitHub reports no licence (verified). The dataset card in the repository (`README_HF.md`) states `cc-by-sa-3.0`, but the owners' loader carries the comment "Assuming generic open license, verify if possible" next to it (verified). So CC BY-SA 3.0 is the owners' stated assumption, not a licence file. Our use is research; no image is redistributed; only example figures made from a few images would appear in the paper, with attribution and the share-alike notice.
- **Access level:** open, no account (verified: public repository, about 2.1 GB).
- **Citation required:** Mohanty et al., Frontiers in Plant Science 2016;7:1419, doi:10.3389/fpls.2016.01419 (verified in `CITATION.cff`).
- **Content (verified on the file list):**
  - `raw/color`: 54,305 files in 38 class folders (the README says 54,306). `raw/segmented`: 54,306 files, 38 folders, names end in `_final_masked.jpg`. `raw/grayscale` is not used.
  - Twins: 54,304 of the 54,305 colour files have a segmented file with the same identifier in the same class; one Tomato late blight image has none. The colour files of Corn common rust have no prefix before the identifier, so pairing must use the identifier, not the whole name. For 1,428 colour files the identifier is not unique within its class (1,032 of them in Corn healthy, mostly "copy" files).
  - Leaf identifier: `leaf-map.json` (SHA-256 `3b4b2536…a25b`) maps a file identifier to a physical leaf. By the owners' own rule (`plant_village.py`), **41,111 colour images (75.7%) get a leaf identifier, in 7,593 leaves (median 4 images per leaf, largest 33); 13,194 images get none.**
  - **Eight classes have no leaf identifier at all (7,887 images):** the four Corn classes, Grape healthy, Squash powdery mildew, Tomato target spot, Tomato mosaic virus. Six more classes are partly covered (5,307 images without one, 2,619 of them in Tomato yellow leaf curl virus).
  - Apple black rot is mapped through a sheet named "Apple_Frogeye Spot" (621 images, 71 leaves): the same disease under another name; the mapping is used.
  - 21 pairs of colour files are byte-identical (same git object), none across two classes.
- **What we use:** colour images, their segmented twins (for the leaf mask), the class from the folder, the leaf identifier.
- **Consequence for the plan (deviation 1, `analysis_plan.md` section 10, `decisions/002-leaf-id-coverage.md`):** the study uses the **30 classes that have leaf identifiers** (46,418 images). Images of those classes without a leaf identifier (5,307) are used for training only, as the approved plan already says. The split becomes 60/15/25 so that small classes have enough test leaves.
- **Not verified (no image opened yet):** that the two copies are aligned pixel for pixel; that the segmented background is pure black; image sizes; how many masks are poor. Notebook 01 measures each and applies the plan's exclusion rules.
- **Known issues:** background and capture conditions carry class information (Molhi 2026, doi:10.1109/cspa68262.2026.11517900, abstract only): this is the subject of the paper. Masks are the owners' automatic segmentation. Laboratory images only.
- **Changes over time:** a git repository; the commit is pinned.

## 2. PlantDoc (exploratory accuracy table only; first to be cut)
- **Name:** PlantDoc, cropped classification set.
- **Owner:** D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, N. Batra (IIT Gandhinagar).
- **URL:** https://github.com/pratikkayal/PlantDoc-Dataset (verified). **Version pinned:** commit `5467f6012d78d1c446145d5f582da6096f852ae8`, 2 May 2021 (verified; unchanged since).
- **Licence:** CC BY 4.0 (verified: `LICENSE.txt`, README and GitHub's licence field). **Access:** open, about 1 GB (verified).
- **Citation:** Singh et al., CoDS-COMAD 2020, doi:10.1145/3371158.3371196 (verified in the README; DOI not yet resolved by us).
- **Content (verified on the file list):** 2,578 image files, 2,342 in `train` (28 class folders) and 236 in `test` (27 folders); the README's abstract says 2,598. One class has 2 images.
- **What we use:** images of the PlantDoc classes that map to one of the 30 PlantVillage classes kept (mapping table fixed in notebook 01 before any model is trained), for accuracy only. The Corn, Squash, Tomato mosaic and healthy-Grape folders drop out with deviation 1.
- **Known issues:** internet images with reported label noise (from memory, not verified); no leaf masks; no leaf or plant identifier, so near-duplicate images cannot be ruled out.

## How the data will be handled
- **Where:** on this PC, under the human's order of 2026-11-19 for T4-class jobs (plan section 10). Notebook `01_audit_masks_split_CPU` clones both repositories at the pinned commits into `projects/qpain/paper-03/data/raw/`, which is ignored by git and read-only afterwards. No approved-access data is involved.
- **Integrity:** the commit is read at the start and the end of the download; SHA-256 of every image used goes to `data/raw/MANIFEST.md` and `outputs/`.
- **Outputs of notebook 01 (derived metadata only):** pairing and mask audit, exclusion counts by class, leaf groups, near-duplicate audit, split file with SHA-256, attribution subset list, PlantDoc mapping table.
- **Privacy and ethics:** leaf photographs; no human or animal data; no ethics approval needed.

## One-time steps for the human
- None to get the data: no account, terms or approval.
- One decision when you are back: the PlantVillage licence is stated only on the dataset card. I treat it as CC BY-SA 3.0 with attribution; say if you want the owners asked.

## Verified on the files by notebook 01 (added 2026-11-22, runs `logs/local-run-01` to `-03`)
- Both repositories fetched at the pinned commits; the remote heads were unchanged before and after. All 46,418 colour images of the 30 classes are 256 x 256.
- The segmented copy is registered with the colour image but **has different colours** (no pixel identical inside the mask in the pairs compared; median absolute difference 18 grey levels). The mask is therefore taken from it, and nothing else. Registration is tested on fine texture (plan deviation 3): 2 images fail.
- Exclusions: 86 of 46,418 (mask above 95% of the image: 66; below 5%: 14; twin ambiguous: 4; misaligned: 2). Usable: 46,332. Table by class: `outputs/pv_exclusions_by_class.csv`.
- Mask area share: median 0.455 (1st to 99th percentile 0.17 to 0.67); it changes by 0.006 on average when the threshold is 8 or 32 instead of 16 (`outputs/pv_mask_threshold_sensitivity.csv`).
- Seen on the overlay sheets (`outputs/mask_overlay_sheet_*.jpg`): the leaf's cast shadow lies outside the mask; some masks miss dark or glossy parts of a leaf or include a strip of shadow; a few images have a black background.
- Stored copies: 14 pairs of images with an identical perceptual hash, all without a leaf identifier; merged (decision 008). Look-alike pairs at larger distances are different leaves and are not merged; their counts by partition pair are in `outputs/near_duplicate_candidates_by_partition.csv`.
- Split (frozen, `notebooks/p3_frozen.json`): 24,716 training images, 5,253 training-only (no leaf identifier), 6,128 validation, 10,235 test; 12,832 leaf groups; attribution subset 785 images (9 to 30 per class).
- PlantDoc: 2,578 files, all readable; 24 files are byte-identical copies of another file; 1,949 images in 22 folders map to a class used. Not scored by any model (plan deviation 4).
- Two of the 50 kept images with the lowest registration value (`outputs/mask_overlay_lowest_alignment.jpg`) have a wrong mask: one is a rectangle around the leaf, one covers only part of the leaf. Such masks pass the area rule. They are kept (the split was frozen before this was seen, and no rule was added afterwards); their share in the whole set is not known beyond these sheets. The paper states that masks are automatic and not corrected.
