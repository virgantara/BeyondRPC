# AdaDCGRL: Adaptive Contrastive Pretraining and Curve-Graph Fusion for Robust Point Cloud Understanding

AdaDCGRL is a two-stage framework that combines **contrastive pretraining using AdaCrossNet with DGCNN** and **structural-aware fine-tuning using DCGRL**, designed for 3D point cloud classification under corruption. This project supports training on clean data and evaluation on corrupted data (PointCloud-C).

## 🔍 Key Contributions

- ✅ **Adaptive Contrastive Learning**: Intra-modal and cross-modal contrastive losses with dynamic EWMA weighting.
- ✅ **Pretraining with DGCNN**: Using RGB or grayscale renderings as the secondary modality.
- ✅ **Downstream with DCGRL**: A dual-branch curve-graph architecture for robust structural representation.
- ✅ **Evaluation on Corrupted Data**: Includes PointCloud-C evaluation on ModelNet40 and ShapeNetPart.

---

## 🗂 Project Structure

