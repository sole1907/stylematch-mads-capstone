# DeepFashion CAP Annotation File Schema

How to read the six annotation files in `data/raw/deepfashion/`. See
`DATA_ACCESS.md` for where these files come from and their license terms,
this file is about what's inside them.

## Category files

### `list_category_cloth.txt`: the 50 category vocabulary

```
50
category_name  category_type
Anorak         1
Blazer         1
Blouse         1
...
Capris         2
...
Caftan         3
```

Line 1: total category count. Line 2: column headers. Each line after
that is one category, in a fixed order that line position 1 to 50 refers
back to.

- **`category_name`**: the garment name, e.g. "Blazer," "Capris," "Dress."
- **`category_type`**: which of three body regions the garment belongs to.
  Not the garment name, a coarse grouping of it.
  - `1` = upper-body (20 categories, e.g. Anorak, Blazer, Blouse)
  - `2` = lower-body (14 categories, e.g. Capris, Jeans, Skirt)
  - `3` = full-body (16 categories, e.g. Caftan, Dress, Jumpsuit)

### `list_category_img.txt`: per-image category label

```
289222
image_name  category_label
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg    3
```

Line 1: total image count. Line 2: column headers. Each line after that
is one image.

- **`category_label`**: an integer 1 to 50, referring to the line
  position of a name in `list_category_cloth.txt` (not the
  `category_type` column). `3` here means line 3 of the vocabulary file,
  "Blouse," which matches the filename above.

---

## Attribute files

### `list_attr_cloth.txt`: the 1000 attribute vocabulary

```
1000
attribute_name  attribute_type
a-line                 3
abstract               1
...
acid                   2
...
arrow collar           4
...
americana              5
```

Same two header lines as the category file, then one attribute per line.

- **`attribute_name`**: the attribute itself, e.g. "a-line," "acid,"
  "arrow collar."
- **`attribute_type`**: which of five kinds of attribute this is.
  - `1` = texture related (156 attributes)
  - `2` = fabric related (218 attributes)
  - `3` = shape related (180 attributes)
  - `4` = part related (216 attributes)
  - `5` = style related (230 attributes)

### `list_attr_img.txt`: per-image attribute labels

```
289222
image_name  attribute_labels
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg    -1 -1 1 -1 ... -1 1 -1
```

Line 1: total image count. Line 2: column headers. Each line after that
is one image, followed by exactly 1000 space separated values, one per
attribute in `list_attr_cloth.txt`'s order.

- Each value is `1` (image has this attribute) or `-1` (does not), not
  the more common `0`/`1` encoding. Confirmed directly against the
  downloaded file, only `-1` and `1` appear.
- Unlike `category_label`, this is multi-label, a single image can have
  any number of its 1000 attribute slots set to `1` at once.

---

## Evaluation partition file

### `list_eval_partition.txt`: the official train/val/test split

```
289222
image_name  evaluation_status
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg    train
img/Sheer_Pleated-Front_Blouse/img_00000003.jpg    val
```

Same two header lines, then one line per image with its assigned split
(`train`, `val`, or `test`). Using this file's assignments rather than a
custom random split keeps results comparable to anything else benchmarked
on the same official partition.

---

## Quick reference: which number means what

| Value you're looking at | Comes from | What it actually encodes |
|---|---|---|
| `category_type` column in `list_category_cloth.txt` | 1, 2, or 3 | body region: upper / lower / full |
| `category_label` column in `list_category_img.txt` | 1 to 50 | line position in `list_category_cloth.txt`, i.e. the specific garment name |
| `attribute_type` column in `list_attr_cloth.txt` | 1 to 5 | attribute kind: texture / fabric / shape / part / style |
| each value in `attribute_labels` in `list_attr_img.txt` | -1 or 1 | whether that specific attribute (by position, matching `list_attr_cloth.txt`'s order) applies to this image |
