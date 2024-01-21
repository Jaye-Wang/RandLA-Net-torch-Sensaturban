import numpy as np
from sklearn.neighbors import KDTree
from time import time

if __name__ == '__main__':
    pts = np.random.randn(10, 3)
    print("pts:", pts)
    t1 = time()
    tree = KDTree(pts)
    print("kdtree build time:", time() - t1)
    select_pts_idx = np.random.choice(len(pts), 1)
    select_pts = pts[select_pts_idx, :].reshape(1, -1)
    k_idx = tree.query(select_pts, k=5)
    print(k_idx)
    # pts_rebuild_copy = np.array(tree.data)
    # pts_rebuild_nocopy = np.array(tree.data, copy=False)
    # pts_rebuild_copy[0] = [1, 1, 1]
    # print("pts_1:", pts)
    # pts_rebuild_nocopy[1] = [1, 1, 1]
    # print("pts_2:", pts)