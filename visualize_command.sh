#1つ目のターミナル
source ~/ros2_ws/install/setup.bash ;
ros2 run robot_state_publisher robot_state_publisher   --ros-args -p robot_description:="$(cat /home/kotaro/my_projects/robotics/maniskill_exp/robotagents/assets/xarm7/xarm7_1305_left.urdf)"

#2つ目のターミナル
source ~/ros2_ws/install/setup.bash ;
ros2 run joint_state_publisher_gui joint_state_publisher_gui   --ros-args -p robot_description:="$(cat /home/kotaro/my_projects/robotics/maniskill_exp/robotagents/assets/xarm7/xarm7_1305_left.urdf)"

#3つ目のターミナル
source ~/ros2_ws/install/setup.bash ;
ros2 run rviz2 rviz2 -d ~/ros2_ws/src/xarm_ros2/xarm_description/rviz/display.rviz




#1つ目のターミナル
source ~/ros2_ws/install/setup.bash ;
ros2 run robot_state_publisher robot_state_publisher   --ros-args -p robot_description:="$(cat /home/kotaro/my_projects/robot/maniskill_exp/xarm7_1305_left_ball_ee.urdf)"

#2つ目のターミナル
source ~/ros2_ws/install/setup.bash ;
ros2 run joint_state_publisher_gui joint_state_publisher_gui   --ros-args -p robot_description:="$(cat /home/kotaro/my_projects/robot/maniskill_exp/xarm7_1305_left_ball_ee.urdf)"

#3つ目のターミナル
source ~/ros2_ws/install/setup.bash ;
ros2 run rviz2 rviz2 -d ~/ros2_ws/src/xarm_ros2/xarm_description/rviz/display.rviz


