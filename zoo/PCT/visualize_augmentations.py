import argparse
import numpy as np
import open3d as o3d
from data import ModelNet40
from PointWOLF import PointWOLF
import rsmix_provider_viz as rsmix_provider
import matplotlib.pyplot as plt

MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

def to_open3d(pc, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if isinstance(color, str):
        color_rgb = np.array(o3d.utility.Vector3dVector([plt.cm.get_cmap('tab10')(0)[:3]]))[0]
        color_rgb = np.array(o3d.utility.Vector3dVector([[1, 0, 1]]) if color == 'mediumorchid' else [[1, 0, 0]])[0]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color_rgb, (pc.shape[0], 1)))
    else:
        color_map = {'mediumorchid': [1.0, 0.4, 0.8], 'red': [1.0, 0.0, 0.0]}
        pcd.colors = o3d.utility.Vector3dVector([color_map[c] for c in color])
    return pcd

def main():
    class Args:
        w_num_anchor = 4
        w_sample_type = 'fps'
        w_sigma = 0.5
        w_R_range = 10
        w_S_range = 3
        w_T_range = 0.25
        beta = 1.0
        nsample = 512
        rsmix_prob = 1.0
        knn = True

    args = Args()
    dataset = ModelNet40(num_points=1024, partition='train')

    idx_table = MODELNET40_CLASSES.index('table')
    idx_chair = MODELNET40_CLASSES.index('chair')

    pc_table = pc_chair = None
    for pc, label in dataset:
        if pc_table is None and label == idx_table:
            pc_table = pc
        elif pc_chair is None and label == idx_chair:
            pc_chair = pc
        if pc_table is not None and pc_chair is not None:
            break

    pointwolf = PointWOLF(args)
    _, table_pw = pointwolf(pc_table.copy())
    _, chair_pw = pointwolf(pc_chair.copy())

    pcs_orig = np.stack([pc_table.copy(), pc_chair.copy()])
    labels_orig = np.array([idx_table, idx_chair])
    pcs_rsmix, _, _, _, lam_mask = rsmix_provider.rsmix(
        pcs_orig.copy(), labels_orig,
        beta=args.beta, n_sample=args.nsample, KNN=args.knn, return_index=True
    )
    pc_rsmix = pcs_rsmix[0]
    lam_mask = lam_mask[0]
    colors_rsmix = ['mediumorchid' if x == 0 else 'red' for x in lam_mask]

    _, table_pw2 = pointwolf(pc_table.copy())
    _, chair_pw2 = pointwolf(pc_chair.copy())
    pcs_pw = np.stack([table_pw2.copy(), chair_pw2.copy()])
    pcs_wolfmix, _, _, _ = rsmix_provider.rsmix(
        pcs_pw.copy(), labels_orig,
        beta=args.beta, n_sample=args.nsample, KNN=args.knn
    )
    pc_wolfmix = pcs_wolfmix[0]

    # Create Open3D PointCloud objects
    geometries = [
        to_open3d(pc_table, 'mediumorchid'),
        to_open3d(pc_chair, 'red'),
        to_open3d(pc_rsmix, colors_rsmix),
        to_open3d(table_pw, 'mediumorchid'),
        to_open3d(chair_pw, 'red'),
        to_open3d(pc_wolfmix, 'mediumorchid')
    ]

    print("Showing all point clouds one by one. Press 'q' to close each window.")
    for geo, title in zip(geometries, [
        "table", "chair", "RSMix", "table+PointWOLF", "chair+PointWOLF", "WOLFMix"
    ]):
        print(f"Showing: {title}")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(geo)
        vis.get_render_option().point_size = 15.0
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    main()
