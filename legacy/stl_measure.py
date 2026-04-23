import trimesh

mesh = trimesh.load("xarm7_stick_centered.stl")

# min 座標と max 座標（各軸）
bbox = mesh.bounds    # shape: (2,3)



# BB サイズ
size = bbox[1] - bbox[0]

# バウンディングボックス中心と重心
center = mesh.bounds.mean(axis=0)   # bbox center
centroid = mesh.centroid

print("BB min:", bbox[0])
print("BB max:", bbox[1])
print("Size:", size)
print("Bounding box center:", center)
print("Centroid:", centroid)
