# AdaDCGRL: Adaptive Contrastive Pretraining and Curve-Graph Fusion for Robust Point Cloud Understanding

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyondrpc-a-contrastive-and-augmentation/point-cloud-classification-on-pointcloud-c)](https://paperswithcode.com/sota/point-cloud-classification-on-pointcloud-c?p=beyondrpc-a-contrastive-and-augmentation)

AdaDCGRL is a two-stage framework that combines **contrastive pretraining using AdaCrossNet with DGCNN** and **structural-aware fine-tuning using DCGRL**, designed for 3D point cloud classification under corruption. This project supports training on clean data and evaluation on corrupted data (PointCloud-C).

## 🔍 Key Contributions

- ✅ **Adaptive Contrastive Learning**: Intra-modal and cross-modal contrastive losses with dynamic EWMA weighting.
- ✅ **Pretraining with DGCNN**: Using RGB or grayscale renderings as the secondary modality.
- ✅ **Downstream with DCGRL**: A dual-branch curve-graph architecture for robust structural representation.
- ✅ **Evaluation on Corrupted Data**: Includes PointCloud-C evaluation on ModelNet40 and ShapeNetPart.

---

## Acknowledgements

The zoo code is borrowed from [PointCloud-C](https://github.com/ldkong1205/PointCloud-C) repository.