得られた知見。

### urdfについて
xarm7_1305_left.urdf は実際に研究室にあるxArm7のロボットシリアルナンバーから作成したURDFファイルです。
作成コマンド：
xacro  src/xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro   dof:=7 robot_type:=xarm robot_sn:=XS130508D43A0C   add_gripper:=true add_realsense_d435i:=true  
 -o xarm7_1305_left.urdf

研究室にある右手のurdfも作成しましたが、rightと全く同じになりました。
左手側xArm7_1305のRobotSNはXS130508D43A0Cです。
右手側xArm7_1305のRobotSNはXS130508D43A11です。



### Xarm7の公式URDFインポートの際の暴れだしについて。
背後になにか大きな要因が１つだけあるのかもしれないが、現在は以下の２つの要因からなる。

1. グリッパーのPDゲインとFMAXの設定(かなり繊細で小さいほど安定)
2. グリッパーは、baseパーツ１つと。グリッパーを構成する６つのパーツからなる。６つのパーツに関してはすべて左右対称なので、固有のパーツは３種類。なかで、悪さをしているのは、inner_knockleというパーツで、これをleftとrightでcollisionをコメントアウトすることで、現在は正常に動作中。



### dualarm環境
・agent[0]は右手
・agent[1]は左手



