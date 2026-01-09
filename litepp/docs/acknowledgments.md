# Acknowledgments

LITE++ builds upon several foundational works in multi-object tracking and object detection. We gratefully acknowledge the following contributions:

## Core Foundations

### LITE: Lightweight Integrated Tracking-Feature Extraction

Our work extends the LITE paradigm introduced in:

```bibtex
@inproceedings{juraev2024lite,
  title={LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration},
  author={Juraev, Jumabek and Kim, Hyeonwoo and Kim, Kwanghoon and Park, Dongwook and Lee, Kwangki},
  booktitle={International Conference on Neural Information Processing (ICONIP)},
  year={2024}
}
```

The LITE paradigm demonstrated that appearance features can be efficiently extracted from intermediate layers of detection networks, eliminating the need for separate ReID model inference.

### DeepSORT

The tracking-by-detection paradigm with appearance features:

```bibtex
@inproceedings{wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paber, Dietrich},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2017}
}
```

### YOLOv8 / Ultralytics

Object detection backbone:

```bibtex
@software{jocher2023yolov8,
  title={Ultralytics YOLOv8},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}
```

## Related Works

### StrongSORT

Enhanced DeepSORT with stronger appearance features:

```bibtex
@article{du2023strongsort,
  title={StrongSORT: Make DeepSORT Great Again},
  author={Du, Yunhao and others},
  journal={IEEE Transactions on Multimedia},
  year={2023}
}
```

### ByteTrack

Association strategy that inspired our threshold adaptation:

```bibtex
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and others},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

### HOTA Metric

Evaluation framework used in our experiments:

```bibtex
@article{luiten2021hota,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and others},
  journal={International Journal of Computer Vision},
  year={2021}
}
```

## Datasets

We evaluate on the following benchmarks:

- **MOT17/MOT20**: Milan et al., "MOT16: A Benchmark for Multi-Object Tracking"
- **DanceTrack**: Sun et al., "DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion"

## Software Dependencies

- PyTorch: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
- OpenCV: Bradski, "The OpenCV Library"
- NumPy: Harris et al., "Array programming with NumPy"

## Funding

[Add funding acknowledgments here]

---

We thank all authors of the above works for making their code and models publicly available, enabling reproducible research in multi-object tracking.
