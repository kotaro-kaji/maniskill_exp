# STL Repair And Xarm Finger Visual Notes

## Context

`xarm_description/meshes/gripper/xarm/right_finger.stl` は Bambu Studio の Boolean で自己交差エラーになった。
`xarm_description/meshes/gripper/bio/right_finger.stl` ではなく、実際に必要だったのは `xarm` 側だった。

`right_kirisute1.stl` と `left_kirisute.stl` を `robotagents/assets/xarm7/xarm7_1305_left.urdf` の gripper finger visual として確認した。
最初は `robotagents/assets/` 直下に置いたが、最終的には `package://xarm_description/...` を使うため `xarm_description/meshes/gripper/xarm/` へ move した。
ただし ROS2/RViz では `xarm_description` が `colcon build` と `source install/setup.bash` 済みでないと `package://xarm_description/...` を解決できない。
`ros2 pkg prefix xarm_description` が `Package not found` の場合は、RViz用URDFでは絶対 `file:///...` を使うのが早い。

## STL repair attempts

元の `xarm` finger STL は `trimesh` 上では `watertight=True` かつ `winding_consistent=True` だが、MeshLab 判定では自己交差面が多く、Bambu Studio の Boolean では失敗した。

失敗または不十分だった方法:

- `trimesh` の duplicate/degenerate face cleanup と normal fix は形状を保つが、Bambu の自己交差エラーには効かなかった。
- `pymeshfix.MeshFix.repair()` や `intersection_removal()` は強すぎて、right finger の大部分が削れて形状が崩れた。
- MeshLab boolean union は自己交差面を 0 にできる場合があったが、Bambu Studio ではまだ自己交差扱いになることがあった。
- 500um/350um voxel remesh は見た目は良いが、ファイルサイズと面数が増える。

採用した方法:

```bash
uv run --with pymeshlab --with scikit-image python - <<'PY'
from pathlib import Path
import trimesh
import pymeshlab

src = Path("xarm_description/meshes/gripper/xarm/right_finger.stl")
dst = src.with_name("right_finger_voxel_repaired_1000um.stl")
pitch = 0.001

mesh = trimesh.load_mesh(src, force="mesh")
vox = mesh.voxelized(pitch=pitch).fill()
remesh = vox.marching_cubes
remesh.apply_transform(vox.transform)
trimesh.repair.fix_normals(remesh)
remesh.export(dst)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(str(dst))
ms.apply_filter("compute_selection_by_self_intersections_per_face")
print(ms.current_mesh().selected_face_number())
PY
```

left finger も同じ 1000um voxel remesh で作成した。

残すファイル:

- `xarm_description/meshes/gripper/xarm/right_finger_voxel_repaired_1000um.stl`
- `xarm_description/meshes/gripper/xarm/left_finger_voxel_repaired_1000um.stl`

1000um 以外の voxel remesh と、中間生成の aggressive/meshlab/self repaired STL は削除してよい。

## Size notes

right finger の比較:

- 元STL: 約 0.23 MB、4852 faces、約 17.04 cm^3
- 1000um voxel: 約 0.72 MB、15004 faces、約 20.58 cm^3
- 500um voxel: 約 2.76 MB、57800 faces、約 18.66 cm^3
- 350um voxel: 約 5.64 MB、118192 faces、約 18.42 cm^3

Bambu Studio で通すだけなら 1000um が軽くて十分だった。

## URDF visual replacement

`right_kirisute1.stl` と `left_kirisute.stl` は寸法が `21 x 31 x 60` で、mm単位のSTLだった。
URDF/Sapien ではメートルとして解釈されるため、scale を入れないと 1000倍で表示される。

`robotagents/assets/xarm7/xarm7_1305_left.urdf` で visual mesh に使う場合:

```xml
<mesh filename="file:///home/kotaro/my_projects/robotics/maniskill_exp/xarm_description/meshes/gripper/xarm/right_kirisute1.stl" scale="0.001 0.001 0.001"/>
```

right finger の visual には `right_kirisute1.stl` を使った。
left finger の visual は後で `left_kirisute.stl` に差し替えた。
どちらも mm 単位STLなので、URDFでは `scale="0.001 0.001 0.001"` を付ける。
collision は box のままにした方が、見た目確認でシミュレーションを壊しにくい。

現在の visual mesh 参照:

```xml
<mesh filename="file:///home/kotaro/my_projects/robotics/maniskill_exp/xarm_description/meshes/gripper/xarm/left_kirisute.stl" scale="0.001 0.001 0.001"/>
<mesh filename="file:///home/kotaro/my_projects/robotics/maniskill_exp/xarm_description/meshes/gripper/xarm/right_kirisute1.stl" scale="0.001 0.001 0.001"/>
```

## Correct environment for visual check

`MyDualBoxRotation-v0` は `xarm7_ball_ee` を使うため、`robotagents/assets/xarm7/xarm7_1305_left.urdf` の確認には向かない。

このURDFを使う軽い確認環境:

```bash
uv run python quick_visualization/test2_gpu.py --env-id MyXarm7PickCube-v1
```

`MyXarm7PickCube-v1` は `robotagents/my_xarm7.py` の `my_xarm7` を使い、そこから `robotagents/assets/xarm7/xarm7_1305_left.urdf` が読まれる。

注意:
この環境では `render_mode=human` のviewer windowが作れない状態だと、環境作成後に `AttributeError: 'NoneType' object has no attribute 'should_close'` で落ちることがある。ログに `Environment ID: MyXarm7PickCube-v1` と observation/action space が出ていれば、少なくとも環境作成とURDF読み込みまでは通っている。

## Kirisute visual alignment

`right_kirisute1.stl` は元の `right_finger.stl` から原点がずれていた。
回転のずれはない前提で、元 `right_finger.stl` の表面点群と、`right_kirisute1.stl` を `scale=0.001` した表面点群を比較し、並進だけを最適化した。

採用した right finger visual の並進:

```xml
<origin rpy="0 0 0" xyz="-0.010475 -0.005449 -0.006592"/>
```

採用した left finger visual の並進:

```xml
<origin rpy="0 0 0" xyz="-0.010518 -0.025412 -0.006599"/>
```

点群最近傍距離の改善:

- origin なし: median 約 1.67 mm、p90 約 7.92 mm
- bbox center合わせ: median 約 0.67 mm、p90 約 2.73 mm
- 最適化後: median 約 0.57 mm、p90 約 1.53 mm

left finger の点群最近傍距離の改善:

- origin なし: median 約 20.53 mm、p90 約 25.55 mm
- bbox center合わせ: median 約 0.66 mm、p90 約 2.76 mm
- 最適化後: median 約 0.54 mm、p90 約 1.44 mm

注意:
RViz向けに `file:///home/...` をURDFへ書くと、ROS2/RVizでは読みやすいが、ManiSkill/SapienのURDF loaderでは `file:///...` をそのままパスとして扱い、`cannot make canonical path` で失敗することがある。
