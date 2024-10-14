# Berkeley Humanoid Robot Description (URDF)
## Overview

This package contains a robot description (URDF) of [Berkeley Humanoid](https://berkeley-humanoid.com) developed by [Hybrid Robotics
](https://hybrid-robotics.berkeley.edu/).


[![Berkeley Humanoid Robot Description](doc/thumbnail.png)](doc/thumbnail.png)

## License

This software is released under a [BSD 3-Clause license](LICENSE).

## Usage

Load the Berkeley Humanoidescription to the ROS parameter server:

    roslaunch berkeley_humanoid_description load.launch

To visualize and debug the robot description, start the standalone visualization (note that you have to provide the following additional dependencies: `joint_state_publisher`, `joint_state_publisher_gui` ,`robot_state_publisher`, `rviz`):

    roslaunch berkeley_humanoid_description standalone.launch

### Launch files

* **`load.launch`:** Loads the URDF to the parameter server. Meant to be included in higher level launch files.

* **`standalone.launch`:** A standalone launch file that starts RViz and a joint state publisher to debug the description.

## FAQ
**Q: Why doesn't the maximum torque of each joint match the values in the paper?**

**A:** The maximum torque is limited for safety reasons.

**Q: Where is the arm?**

**A:** This version of the robot is focused on locomotion research. We are currently developing and testing the arm.
