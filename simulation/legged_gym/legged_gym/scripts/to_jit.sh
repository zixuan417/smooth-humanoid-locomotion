# Example:
# bash to_jit.sh gr1 aug22-test

robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_walk_phase"

proj_name="${robot_name}_walk_phase"
exptid=${2}

# Run the training script
python save_jit.py --robot "${robot_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                # --checkpoint 10000 \
