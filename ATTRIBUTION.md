# Attribution, licensing, and project scope

## Project scope

This is a **proof of concept**. It is built for a class presentation on AI ×
cybersecurity, and possibly for research work at NCITE. It is **not a
commercial product and will not become one in its current form.**

That scope is the reason every licence below is fine. Read the last section
before that ever changes.

---

## What to cite

### The dataset — DUT Anti-UAV

> Jie Zhao, Jingshu Zhang, Dongdong Li, Dong Wang.
> "Vision-based Anti-UAV Detection and Tracking."
> *IEEE Transactions on Intelligent Transportation Systems*, 2022.
> arXiv:2205.10851 · https://github.com/wangdongdut/DUT-Anti-UAV

10,000 images across train (5,200) / val (2,600) / test (2,200), annotated in
Pascal VOC XML, single class `UAV`.

> **Verify before you submit anything:** I could not reach IEEE Xplore or the
> arXiv API from this machine, so the volume, issue, and page numbers are not
> filled in above. Get them from IEEE Xplore and complete the citation. The
> author list, title, venue and year come from the authors' own repository, so
> those are reliable.

### The background images — COCO 2017

> Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona,
> Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick.
> "Microsoft COCO: Common Objects in Context." *ECCV*, 2014.
> https://cocodataset.org

417 drone-free images selected from COCO 2017. COCO **annotations** are
CC BY 4.0; the **images** are individually licensed, and every one is recorded
in `data_negatives/manifest.csv` with its licence and original URL. If your
presentation shows any of these images individually, attribute that image using
its manifest row.

### The model — Ultralytics YOLO

> Ultralytics YOLO (version 8.4.145), AGPL-3.0.
> https://github.com/ultralytics/ultralytics

Cite the specific YOLO11 model variant you actually train (`yolo11n` etc.) and
use Ultralytics' own citation text from their repository, so the version and
authorship are right at the time you present.

---

## Licence status

| Component | Licence | OK for this project? |
|---|---|---|
| Ultralytics YOLO 8.4.145 | AGPL-3.0 | **Yes** — running it for research and teaching is unrestricted |
| `yolo11n.pt` pretrained weights | AGPL-3.0 | **Yes**, same reason |
| DUT Anti-UAV images | **None stated** | Yes for academic use, with citation |
| DUT GitHub repository | Apache-2.0 | Yes (covers the repo, not the hosted images) |
| COCO annotations | CC BY 4.0 | Yes |
| COCO images (169 of 417) | CC BY 2.0 / CC BY-SA 2.0 | Yes, with attribution |
| COCO images (248 of 417) | CC BY-NC / CC BY-NC-SA 2.0 | **Yes — non-commercial use only** |
| The four scripts in `code/` | Yours | Yes, no inheritance |

**AGPL and your presentation:** AGPL's obligations trigger on *distribution* —
shipping software, or running it as a network service for other people. Running
it on your own machine, showing results on a slide, or live-demoing it in class
triggers nothing. If you put a web demo online for others to use, you would owe
those users the source; the simple answer there is to publish your scripts,
which you can do freely since they are yours.

**The DUT "no licence" issue:** unstated terms are not the same as permissive
terms. For academic use with proper citation this is the normal, accepted
practice. It is only a problem for a product.

---

## If this ever does go commercial

The plan is to rebuild rather than retrofit, which is the right order. Three
things would have to change, in descending difficulty:

1. **Ultralytics AGPL-3.0.** Either buy their Enterprise Licence, or rebuild on
   a permissively licensed detection stack. This applies regardless of what data
   you use, and it is the largest obstacle.
2. **The DUT dataset.** Get written permission from the authors at Dalian
   University of Technology, or replace it with footage you own.
3. **The 248 NonCommercial negatives.** Already handled: set
   `COMMERCIAL_SAFE_ONLY = True` in `code/add_negatives.py` and it drops to the
   169 permissively licensed images automatically.

The contamination lives in the **trained weights**, not the code. A model
trained on non-commercially-licensed data is plausibly a derivative of it, and
you cannot fix that by continuing to train from the same checkpoint. A
commercial version needs a clean retrain from a clean base.

**What carries forward regardless:** all four scripts in `code/`, the whole
conversion and verification pipeline, everything you learn about what accuracy
this problem actually needs, and the knowledge of which camera, resolution, and
image size are required to detect a drone at the range you care about. That
last part is the expensive knowledge, and a proof of concept is exactly how you
buy it cheaply.

*None of this is legal advice. If commercialisation becomes real, have a
lawyer look at the AGPL question specifically — it is the one that surprises
people.*
