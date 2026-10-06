# Literature — vision-based counter-UAS

12 PDFs in `pdf/`, all legitimately free (arXiv preprints and CVF open access).
Every file was verified after download by extracting its text and matching the
title — none is a wrong or broken file.

Paywalled papers are listed at the bottom with links, not downloaded. Get those
through the UNO library; do not use shadow-library sites.

## Read these three first

1. **`01_DUT_AntiUAV_2022.pdf`** — your dataset's own paper. Section IV and
   Table II are the numbers you are reproducing. Read this one properly.
2. **`13_IsaacMedina_benchmark_2021.pdf`** — the competing benchmark. Compares
   Faster R-CNN / YOLOv3 / SSD / DETR across three UAV datasets. This is the
   paper DUT is implicitly arguing against.
3. **`02_AntiUAV_benchmark_2021.pdf`** — the large multi-modal (RGB + infrared)
   benchmark. The reason the "is RGB enough?" question is live.

## The reading list, grouped the way the synthesis groups them

### A. Is vision alone sufficient? (the central disagreement)

| File | Paper |
|---|---|
| `01_DUT_AntiUAV_2022.pdf` | Zhao, Zhang, Li, Wang. *Vision-based Anti-UAV Detection and Tracking.* IEEE T-ITS, 2022. |
| `02_AntiUAV_benchmark_2021.pdf` | Jiang et al. *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking.* arXiv:2101.08466, 2021. |
| `03_AntiUAV_2nd_workshop_2021.pdf` | Zhao et al. *The 2nd Anti-UAV Workshop & Challenge: Methods and Results.* arXiv:2108.09909, 2021. |

DUT argues RGB vision is now mature enough on its own. The Anti-UAV benchmark
and workshop are built on RGB **plus infrared** video pairs — a design choice
that implicitly disputes it.

### B. What makes it hard — two different framings

| File | Paper |
|---|---|
| `13_IsaacMedina_benchmark_2021.pdf` | Isaac-Medina, Poyser, Organisciak, Willcocks, Breckon, Shum. *UAV Visual Detection and Tracking using Deep Neural Networks: A Performance Benchmark.* ICCV Workshops, 2021, pp. 1223–1232. |
| *(paywalled, see below)* | Coluccia et al. *Drone-vs-Bird Detection Challenge at IEEE AVSS 2019.* |
| *(paywalled, see below)* | Wang et al. *Flying Small Target Detection for Anti-UAV…* Sensors 19(9):2168, 2019. |

DUT frames the problem as **small-target detection against complex
backgrounds**. Drone-vs-Bird frames it as **discriminating drones from birds** —
a false-positive problem. Different framings produce different metrics, and
that is a gap worth naming.

### C. The detectors being compared

| File | Paper |
|---|---|
| `05_CascadeRCNN_2018.pdf` | Cai, Vasconcelos. *Cascade R-CNN.* CVPR 2018. **Best on DUT: mAP 0.683** |
| `06_FasterRCNN_2015.pdf` | Ren, He, Girshick, Sun. *Faster R-CNN.* NeurIPS 2015. |
| `08_ATSS_2020.pdf` | Zhang et al. *ATSS.* CVPR 2020. |
| `04_YOLOX_2021.pdf` | Ge et al. *YOLOX.* arXiv:2107.08430, 2021. **Fastest on DUT: 53.7 FPS** |
| `07_SSD_2016.pdf` | Liu et al. *SSD.* ECCV 2016. |
| `10_YOLOv3_2018.pdf` | Redmon, Farhadi. *YOLOv3.* arXiv:1804.02767, 2018. |
| `09_DETR_2020.pdf` | Carion et al. *DETR.* ECCV 2020. |

You do not need to read all seven. Read Cascade R-CNN and YOLOX — they are the
two ends of the accuracy/speed tradeoff in DUT's own Table II, and that
tradeoff is the point.

### D. Background

| File | Paper |
|---|---|
| `11_COCO_2014.pdf` | Lin et al. *Microsoft COCO.* ECCV 2014. Source of your 417 negatives, and of the pretrained weights. |

## Paywalled — get these through the UNO library

These are genuinely relevant but not freely available. Not downloaded.

| Paper | Where |
|---|---|
| Coluccia et al. *Drone-vs-Bird Detection Challenge at IEEE AVSS 2019.* | IEEE Xplore |
| Shi, Yang, Xie, Liang, Shi, Chen. *Anti-drone system with multiple surveillance technologies.* IEEE Communications Magazine 56(4), 2018. | IEEE Xplore |
| Hu, Wu, Zheng, Liu. *Object detection of UAV for anti-UAV based on improved YOLOv3.* Chinese Control Conference, 2019. | IEEE Xplore |
| Hoffmann et al. *Micro-Doppler based detection and tracking of UAVs with multistatic radar.* IEEE Radar Conf, 2016. | IEEE Xplore |
| Abunada et al. *Design and implementation of an RF based anti-drone system.* IEEE ICIoT, 2020. | IEEE Xplore |
| Chang et al. *A surveillance system for drone localization and tracking using acoustic arrays.* IEEE SAM, 2018. | IEEE Xplore |
| Wang et al. *Flying Small Target Detection for Anti-UAV…* Sensors 19(9):2168, 2019. | Open access at mdpi.com — their server blocks scripted download, so open it in a browser: https://www.mdpi.com/1424-8220/19/9/2168 |

The three IEEE sensing papers (radar / RF / acoustic) matter for one paragraph
of the synthesis only — you need them to characterise the "vision is not
enough" position fairly. The abstracts may be sufficient for that.

## Provenance

All citations were taken from the bibliography of `01_DUT_AntiUAV_2022.pdf`
(extracted from the PDF), not from memory. Where an arXiv identifier was
guessed rather than known, the downloaded file was verified by matching the
title inside the PDF. One guessed identifier was wrong and was discarded rather
than cited.
