# yahboom_ws_peter

Mirror of the **vendor + community** ROS 2 packages from the cat-patrol
robot's workspace, *excluding* the two custom packages we maintain
separately (`cat_patrol_robot/` and `odom_drift_checker/`).

This repo exists so that a fresh checkout of the project is reproducible:
clone this repo, drop it into `your_workspace/src/`, then clone
`cat_patrol_robot/` and `odom_drift_checker/` alongside, and `colcon
build` should produce the same workspace as the live robot.

## Layout

    yahboom_ws_peter/
    └── src/
        ├── yahboomcar_bringup/        ← Yahboom vendor (with our
        │                                 trim-parameter modifications;
        │                                 see cat_patrol_robot/yahboom_overlay
        │                                 for the patch record)
        ├── yahboomcar_base_node/      ← Yahboom vendor (C++ odometry integrator)
        ├── yahboomcar_description/    ← Yahboom vendor (URDFs and meshes)
        ├── yahboomcar_msgs/           ← Yahboom vendor (message types)
        └── ...                        ← all other yahboomcar_* packages
                                          plus laserscan_to_point_pulisher,
                                          robot_pose_publisher_ros2, etc.

## Sync workflow

This repo is populated by running, on the host machine (not the robot):

    /opt/robots/cat_patrol_robot/scripts/sync_robot_to_host.sh

That script rsyncs the live workspace from the Jetson to the host mirror,
then mirrors each package to the right place in `/opt/robots/`.

Push-only: do NOT git-pull into this repo to bring changes back to the
robot. Edits flow robot → host → git, never the other way.
