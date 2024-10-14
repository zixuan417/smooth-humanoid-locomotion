# Example:
# bash run.sh gr1 aug22-test cuda:0
# bash run.sh h1 aug22-test cuda:0
# bash run.sh gr2t4 aug22-test cuda:0

robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_walk_phase"

proj_name="${robot_name}_walk_phase"
exptid=${2}

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${3}" \
                --fix_action_std \
                # --debug
                # --resume \
                # --resumeid xxx
