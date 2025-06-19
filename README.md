# BeyondRPC: A Contrastive and Augmentation-Driven Framework for Robust Point Cloud Understanding 

Beyond RPC is a framework combining 3D Point Cloud Adaptive Contrastive Learning with WOLFMix 

## 🔍 Key Contributions

- ✅ **Adaptive Contrastive Learning**: Intra-modal and cross-modal contrastive losses with dynamic EWMA weighting.
- ✅ **Pretraining with DGCNN**: Using RGB or grayscale renderings as the secondary modality.
- ✅ **Evaluation on Corrupted Data**: Includes PointCloud-C evaluation on ModelNet40 and ShapeNetPart.

## Paper Link
[PDF](https://inass.org/wp-content/uploads/2025/05/2025073142-3.pdf)


> ⚠️ **Correction Note (June 2025):**  
> In the published paper, the value reported as the mean Corruption Error (mCE) was mistakenly computed as the square root of the actual mCE.  
> The correct mCE is **0.59**. This correction does **not affect the ranking between models or the conclusions of the paper.**

---

## Acknowledgements

The zoo code is borrowed from [PointCloud-C](https://github.com/ldkong1205/PointCloud-C) repository.