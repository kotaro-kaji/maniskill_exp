import trimesh

mesh = trimesh.load("xarm7_stick.stl")

# バウンディングボックス中心 or 重心
center = mesh.bounds.mean(axis=0)   # bbox center
# center = mesh.centroid            # 重心を基準にしたい場合

# 平行移動して中心を原点に
mesh.apply_translation(-center)

# 保存
mesh.export("xarm7_stick_centered.stl")
