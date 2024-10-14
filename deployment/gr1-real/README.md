# Deployment Code for Fourier GR1T2 Robot

## Installing Dependencies
Create environment, the python version should be 3.11 as required by the control sdk:
``` bash
conda create -n gr1-real python=3.11 
```

Install control SDK:
   ```bash
   cd Thirdparty/wiki-gr1-python
   python -m pip install robot_rcs-0.4.0.13-cp311-cp311-manylinux_2_30_x86_64.whl
   python -m pip install robot_rcs_gr-1.9.1.14-cp311-cp311-manylinux_2_30_x86_64.whl
   ```

## Preparations
Following will be the steps to prepare for the deployment.

### Convert Simulation PD Gains to Real Robot PD Gains
Please refer to the file [convert_gains.py](./scripts/convert_gains.py) to convert the simulation PD gains to real robot PD gains. Currently, the PD gain used for real robot corresponds to the simulation PD gains in this repo. If you modify the simulation PD gains and want to use them for the real robot, you need to convert them using the script. Please refer to the example in the script for converting, and then write the converted values manually in the file [gr1_robot.py](./scripts/gr1_robot.py).

### Calibrate the Robot
Please refer to the official guide provided by Fourier Intelligence to prepare the real robot. Then run the command:
```bash
cd scripts && python set_home.py
```
The file [sensor_offset.json](./scripts/sensor_offset.json) should contain the sensor offsets. Notice that this step is only required the first time you deploy the robot.


## Deployment
After the preparations, you can start deploy our provided model with command:
```bash
cd scripts && python gr1_deploy_phase.py
```
After training your own model, you can move it to the folder [deploy_models](./deploy_models/), and then specify your model name in the script [gr1_deploy_phase.py](./scripts/gr1_deploy_phase.py), and you are ready to deploy your own model!

### Joystick Control
As this version of sdk does not provide official joystick control api, we implemented a simple joystick control script that can receive X-Box controller signals. Please refer to the script [joystick.py](./utils/joystick.py) for more details. For plugging in the code, you can just set the argument `--use_joystick` to `True` in the script [gr1_deploy_phase.py](./scripts/gr1_deploy_phase.py).


## Known Issues
This version of sdk is not the latest version, which you can find [here](https://pypi.org/project/fourier-grx/). If you are going to use the latest version, the only thing you should rewrite is [gr1_robot.py](./scripts/gr1_robot.py).