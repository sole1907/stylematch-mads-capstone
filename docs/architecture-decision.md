# Architecture decision: similarity-engine backbone

**Decision**: Marqo-FashionCLIP, fine-tuned — not used zero-shot/frozen.

## Options considered

1. **FashionCLIP** (patrickjohncyh/fashion-clip) — CLIP ViT-B/32 fine-tuned for
   fashion product embeddings. Named in the original proposal.
2. **Marqo-FashionCLIP** — successor model, fine-tuned from ViT-B-16
   (laion2b_s34b_b88k) via Generalised Contrastive Learning. Claims +57%
   text-to-image retrieval and +11% category-to-product vs. FashionCLIP 2.0
   on its own leaderboard.
3. **rainalexotl/deepfashion-image-classification** — modular CNN/ResNet/ViT
   pipeline built on DeepFashion, but classification-focused; would require
   reusing its data-loading/backbone and swapping in our own metric-learning
   head.

## Why Marqo-FashionCLIP

- Strict upgrade over the originally-named FashionCLIP on retrieval-relevant
  metrics, same general approach.
- Already produces fashion-domain embeddings — fine-tuning adapts/specializes
  rather than building fashion-relevant representations from scratch, which
  fits a 7-week timeline better than starting from rainalexotl's
  ImageNet-pretrained backbones.
- Apache 2.0 license — permissive, no restriction on our use (include license
  text, note significant changes if redistributed).
- Fine-tuning plan: freeze most of the backbone, unfreeze only the last few
  layers, train a new projection head with `pytorch-metric-learning`
  triplet/contrastive loss on DeepFashion. This is real, demonstrable training
  work — not just invoking a pretrained model zero-shot — while staying
  scope-appropriate for the timeline.

## Known risk: backbone pretraining leakage

Marqo-FashionCLIP was evaluated on **DeepFashion-InShop** and
**DeepFashion-Multimodal** as two of its seven benchmark datasets.

- Our primary training data is the **Category and Attribute Prediction**
  subset — a different DeepFashion subset from the ones Marqo evaluated on.
  Risk is lower here, but we can't fully confirm zero image-level overlap
  across DeepFashion's officially released subsets (they're drawn from the
  same underlying photo corpus) — disclose as a limitation if asked, rather
  than assert clean separation.
- This **further weakens the already-conditional FashionNet/In-Shop retrieval
  comparison** named in the Evaluation Plan. Even setting aside the unresolved
  access question for that subset, our backbone has already seen that exact
  benchmark during Marqo's own training — any retrieval comparison against it
  would be contaminated, not a fair test. Treat that comparison as unlikely
  to happen rather than conditional-but-plausible; lean fully on the naive
  baseline + Vasileva et al. comparison (compatibility engine) as the real
  committed evaluation.

*Drafted 2026-10-21.*
