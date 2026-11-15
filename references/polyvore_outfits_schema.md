# Polyvore Outfits Dataset Schema

How to read the files in `data/raw/polyvore_outfits/`, once sourced. Same
precedent as `deepfashion_schema.md`: use the real, standard dataset
release format rather than inventing our own.

We use the **nondisjoint** split (`polyvore_outfits/nondisjoint/`, not
`disjoint/`) — items can repeat across train/valid/test, but this is the
split Vasileva et al.'s cited numbers (AUC=0.86, FITB=55.3%) are reported
on, so our results stay directly comparable without a split mismatch.

## Directory layout

```
polyvore_outfits/
  images/                          # one file per unique item
    <item_id>.jpg
  polyvore_item_metadata.json      # item_id -> metadata, split-independent
  nondisjoint/
    train.json
    valid.json
    test.json
    compatibility_train.txt
    compatibility_valid.txt
    compatibility_test.txt
    fill_in_blank_valid.json
    fill_in_blank_test.json
```

## `{train,valid,test}.json`: outfits

A list of outfit dicts:

```json
[
  {"set_id": "205000121", "items": [{"item_id": "123456789", "index": 1}, {"item_id": "987654321", "index": 2}]},
  ...
]
```

- **`set_id`**: unique outfit ID.
- **`items`**: ordered list of this outfit's items. **`item_id`** is the
  globally unique ID, matching both `polyvore_item_metadata.json`'s keys
  and `images/<item_id>.jpg`. **`index`**: the item's 1-indexed position
  within this specific outfit (an item can have a different `index` in a
  different outfit it also appears in, in the nondisjoint split).

## `polyvore_item_metadata.json`: per-item metadata

`item_id -> {"semantic_category": ..., "title": ..., "url_name": ..., ...}`

- **`semantic_category`**: the field to use as our category label (the
  compatibility-engine analogue of DeepFashion's `category_label`).
- `title` / `url_name`: human-readable item description, not needed for
  training, useful for debugging/sanity-checking retrieved items.

## `compatibility_{valid,test}.txt`: the AUC eval task

Space-separated, one line per question: `label item_1 item_2 ...
item_n`. **`label`**: `1` = compatible (a real outfit), `0` = incompatible
(a generated negative).

## `fill_in_blank_{valid,test}.json`: the FITB eval task

A list of question dicts, each with a **`question`** list (the outfit's
items minus one, i.e. the context) and an **`answers`** list (candidates
to fill the blank — 4-way per the paper).

## Our dataset interface, independent of the above

`stylematch/compatibility/dataset.py`'s three classes
(`PolyvoreOutfitDataset`, `FITBDataset`, `CompatibilityDataset`) return
plain `(images, labels, ...)` tuples — the exact raw-file parsing details
above are resolved once inside `__getitem__`, not exposed to anything
downstream. Training/model code never needs to know whether an ID was
composite or not.
