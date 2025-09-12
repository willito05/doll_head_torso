# doll_head_torso

Detect if a doll is **assembled** (head+torso) or **disassembled**, and publish the decision plus **2D/3D part coordinates**.

## Quickstart
```bash
# clone
git clone git@github.com:willito05/doll_head_torso.git
cd doll_head_torso

# download weights (and optional dataset if URL set)
./scripts/download_assets.sh

# build inside catkin workspace
cd ~/catkin_ws/src && ln -s $(pwd) .
cd .. && catkin_make && source devel/setup.bash

# run
roslaunch doll_head_torso doll_state_cord.launch

# visualize
rqt_image_view /doll_state/debug_image
rostopic echo /doll_state/state
