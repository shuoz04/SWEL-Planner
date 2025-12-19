# SWEL-Planner
**This is the source code of our proposed SWEL planner**
## overview
![overview](mpv2/output/overview.png)
## robot in real-world
![real_world](mpv2/output/realworld.png)
## config
ubuntu 20.04
ROS ros-noetic
moveit 1.1.16
## Repository Structure
```text
.
├── mpv2/                  # path planning algorithm
│   ├── scripts            # core codes
│   └── condig             # config
├── rm_moveit/             # moveit config
│   ├── launch             # rviz/gazebo launch
│   ├── config
│  
├── rm_description/        # robot file
│   └── meshes
    └── urdf               # URDF file
├── scene_describe/        # task scene modeling
│   ├── GMM
│   ├── scene-modeling
│   └── oct_Tree          
└── README.md
```
## TODO
Providing Unified Interface and document
## Reference
  NIRRT-star：https://github.com/tedhuang96/nirrt_star

