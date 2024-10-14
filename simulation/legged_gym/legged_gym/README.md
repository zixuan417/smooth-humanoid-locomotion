## How to use

### Display reference motion.

```
python scripts/view.py robot=gr1t1_mimic_robot robot.motion.motion_name=02_01 save=false robot.task_name=mimic_view_motion
```

If you want to save key bodies, set save to be true.

### Train exwholebody control

```
python main.py mode=run robot=gr1t1_mimic_robot device=cuda:0 
```

You can change parameters in `envs/{robot_name}/{robot_name}_mimic_config` or run the command as following:

```
python main.py mode=run robot=gr1t1_mimic_robot device=cuda:0 robot.rewards.scales.alive=5 robot.rewards.scales.tracking_demo_dof_pos=3
```

### Display Policy

If you want to save videos, run the following scripts

```
python scripts/eval.py --exp_desc=04-14-15 --robot gr1t1 --task mimic --rl PPO --mode run --eval_task mimic_eval --motion_name motions_debug_gr1t1.yaml --record_video --num_envs 22
```

If you just want to display the results

```
python scripts/eval.py --exp_desc=04-14-15 --robot gr1t1 --task mimic --rl PPO --mode run --eval_task mimic_eval --motion_name motions_debug_gr1t1.yaml
```

*exp_desc is a substring of experiment name in run directory*