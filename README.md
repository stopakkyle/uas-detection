# Vision-Based UAS Detection

Fine-tuning a YOLO object detector on the DUT Anti-UAV dataset, reproducing the
published benchmark, and auditing that benchmark's evaluation hygiene.

Coursework proof of concept — not a commercial system. See `ATTRIBUTION.md` for
citations and licence status.

## Result

On the held-out test split (2,200 images, 2,245 annotated drones, used once):

| Metric | Value |
|---|---|
| mAP50 | **0.929** |
| mAP50-95 | **0.596** |
| Precision | 0.947 |
| Recall | 0.881 |
| Inference | 0.8 ms/image |

YOLO26-nano, 25 epochs, 640px, batch 8, **seed 0** with deterministic ops, on an
Apple M2 laptop GPU. Full settings in `runs/uav_yolo26n/run_record.txt`.

The published reference is 0.683 mAP (Cascade R-CNN + ResNet-50, Zhao et al.
2022, Table II). That paper does not state which IoU threshold its "mAP" uses,
so both of our figures are reported rather than whichever is more flattering.

**Caveat:** a perceptual-hash audit found 64.8% of test images have a
near-duplicate in training — the dataset's official split partitions video
frames, not clips. Our figure, and every published figure on this dataset, is
optimistic as a measure of generalisation to new scenes. See
`literature/Milestone1.docx` §3.

## What's here

```
code/                 the pipeline (read code/README.md first)
runs/uav_yolo26n/     training curves, metrics, trained weights, run record
runs/uav_yolo26n_TEST/ final test-split evaluation
data_negatives/       manifest + provenance for 417 background images
literature/           reading guide and the milestone write-up
ATTRIBUTION.md        citations and licence status
```

Datasets and PDFs are not in the repository — see "Reproducing" below.

## Reproducing

**1. Get the dataset.** Download DUT Anti-UAV from the authors and place it at
`data/{train,val,test}/{img,xml}/`:
<https://github.com/wangdongdut/DUT-Anti-UAV>

**2. Install.**

```bash
pip3 install ultralytics pillow
```

**3. Convert, add negatives, verify.**

```bash
cd code
python3 voc_to_yolo.py              # VOC XML -> YOLO txt, builds dataset_yolo/
python3 add_negatives.py            # mixes in the background images
python3 verify_yolo_annotations.py  # draws boxes back on, for visual checking
```

`add_negatives.py` expects the 417 images in `data_negatives/img/`. They are not
redistributed here — most are CC NonCommercial. `data_negatives/manifest.csv`
lists every image's COCO id, licence and original URL, so the set can be rebuilt
exactly.

**4. Train and evaluate.**

```bash
python3 train_yolo.py      # ~3 hours on an M2; --resume continues an interrupted run
python3 evaluate_test.py   # the final held-out evaluation, run once
```

All paths are set in the CONFIGURATION block at the top of each script.

## Notes

- Conversion is lossless: converted object counts match the paper's Table I
  exactly (5,243 train / 2,245 test instances).
- The model had not converged at 25 epochs — every validation curve was still
  improving — so the reported figure is a lower bound.
- Training and conversion are deliberately separate scripts so that training is
  never triggered as a side effect of preparing data.
