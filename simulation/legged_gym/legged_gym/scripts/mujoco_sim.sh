# Example:
# bash mujoco_sim.sh gr1 aug22-test

robot_name=${1}

proj_name="${robot_name}_walk_phase"
exptid=${2}

# Run the training script
python sim2sim.py --robot "${robot_name}" \
                --exptid "${exptid}" \
                # --record_video \
