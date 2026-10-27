# Data Access Statement

How to access each dataset used in this project, who owns it, and what
restrictions apply.

---

## DeepFashion: Category and Attribute Prediction (CAP) subset

**Used for:** similarity engine training.

- **Source:** http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html (subset
  page: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
- **Owner:** Multimedia Lab, The Chinese University of Hong Kong (Liu et al., 2016)
- **License:** non-commercial research use only. Per the project page's own
  News section, the CAP subset's data and labels are "released without
  encryption (password)." No redistribution beyond internal use, raw
  images are **not** re-hosted in this repo.
- **Access method:** direct download from Google Drive, no account approval
  needed. Six files: `img.zip` (images), `list_attr_cloth.txt` /
  `list_attr_img.txt` (attribute vocabulary / per-image labels),
  `list_category_cloth.txt` / `list_category_img.txt` (category vocabulary
  / per-image labels), `list_eval_partition.txt` (official train/val/test
  split).

### Option 1: automated script

```bash
python scripts/download_deepfashion.py
```

Downloads all six files into `data/raw/deepfashion/` and extracts
`img.zip`. Safe to re-run, already-downloaded files are skipped by
default (`--skip-existing`, on by default). If a run is interrupted or
partially fails, just run it again; it picks up wherever it left off.

**Google Drive throttling:** these files are shared publicly, which means
Google Drive sometimes rate-limits downloads with a "too many users have
viewed or downloaded this file recently" error. The script keeps going if
one file hits this (so a block on the large `img.zip` doesn't stop the
small label files from downloading), and prints a one-line summary at the
end, e.g. `SUMMARY: 4/6 files done so far, rate limit likely blocking the
rest`. If you hit this, just wait and re-run later, it's usually resolved
within an hour or so.

### Option 2: manual download

If the script keeps hitting rate limits, each file can be downloaded by
hand from its Drive link below, and saved into `data/raw/deepfashion/`
under the exact filename listed.

| File | Link |
|---|---|
| `img.zip` | https://drive.google.com/file/d/0B7EVK8r0v71pa2EyNEJ0dE9zbU0/view?usp=drive_link&resourcekey=0-CPiKS-AiE8IDonk54WJ5_w |
| `list_attr_cloth.txt` | https://drive.google.com/file/d/0B7EVK8r0v71pYnBKQVBOaHR1WWs/view?usp=drive_link&resourcekey=0-N8iNXTUvwLteEixb7emjwg |
| `list_attr_img.txt` | https://drive.google.com/file/d/0B7EVK8r0v71pWXE4QWotX2hxQ1U/view?usp=drive_link&resourcekey=0-3w0aVsDrCfaDNG1eH6Pirw |
| `list_category_cloth.txt` | https://drive.google.com/file/d/0B7EVK8r0v71pWnFiNlNGTVloLUk/view?usp=drive_link&resourcekey=0-ClgptLwFKxgFU4HfAakqmw |
| `list_category_img.txt` | https://drive.google.com/file/d/0B7EVK8r0v71pTGNoWkhZeVpzbFk/view?usp=drive_link&resourcekey=0-Dm6R_wuniTptUQSrWT7tSg |
| `list_eval_partition.txt` | https://drive.google.com/file/d/0B7EVK8r0v71pdS1FMlNreEwtc1E/view?usp=drive_link&resourcekey=0-_AL8Vhcl1UUYuw7NbGnZtA |

After a manual `img.zip` download, extract it into
`data/raw/deepfashion/` (should produce an `img/` folder with ~5,600
category subfolders), re-running the script afterward will detect the
extracted folder and skip re-extracting.

For what's actually inside the annotation files (category/attribute
vocabulary, encoding, the train/val/test split), see
[`references/deepfashion_schema.md`](references/deepfashion_schema.md).

---

## Polyvore Outfits

**Used for:** compatibility engine training.

- **Source:** https://huggingface.co/datasets/mvasil/polyvore-outfits
  (paper: https://arxiv.org/abs/1803.09196, original repo:
  https://github.com/mvasil/fashion-compatibility)
- **Owner:** Mariya I. Vasileva et al., 2018
- **License:** gated, requires requesting access on Hugging Face with an
  institutional/research-affiliated email, subject to manual approval.
  Explicitly "intended solely for academic and non-profit research
  purposes." We're requesting access with a university email and treating
  it as academic-use-only; no redistribution.
- **Access method:** request access at the Hugging Face link above
  (institutional email required). Status: access request pending, will be
  updated here once approved and the download pipeline is in place.

---

## AFRIFASHION1600 *(stretch goal only, bias generalization evaluation)*

**Used for:** optional, time permitting evaluation of whether the
similarity engine generalizes to underrepresented African fashion styles.
Not part of the committed MVP, not used for training.

- **Source:** https://github.com/DataScienceNigeria/Research-Papers-by-Data-Science-Nigeria/tree/master/AFRIFASHION1600%3A%20A%20Contemporary%20African%20Fashion%20Dataset%20for%20Computer%20Vision
  (Oyewusi et al., CVPR 2021 Workshop), the official repository of the
  research group behind the paper.
- **Owner:** Data Science Nigeria / Oyewusi et al.
- **License:** no formal license tag. The repository's README states it
  "was curated from openly available images online and is open-sourced,
  but all images belong to the original copyright owners." Treated the
  same way as DeepFashion: academic/research use, proper citation, no
  re-hosting of raw images in this repo.
- **Access method:** direct download from the repository above, no request
  process. Not yet downloaded, only needed if the team reaches the
  stretch goal.
