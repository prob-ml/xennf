import itertools
import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

def format_command(name):
    return (
        f"python nf_potts.py "
        f"--config_name={name}"
    )

# Generate the command for each combination
config_list = [x.replace('config', '').replace('.yaml', '') for x in os.listdir('config/config_SYNTHETIC')]
commands = [format_command(i) for i in range(len(config_list))]

# Split commands into 3 groups for the 3 GPUs
gpu_batches = [commands[i::3] for i in range(3)]

# Create a directory to store the shell scripts
os.makedirs('scripts', exist_ok=True)

# Create a script for each GPU that runs commands sequentially
for gpu_id, batch in enumerate(gpu_batches):
    script_name = f'scripts/gpu_{gpu_id}.sh'
    
    # Write commands to the shell script
    with open(script_name, 'w') as f:
        f.write('#!/bin/bash\n')
        # Set specific GPU for this batch
        f.write(f'export CUDA_VISIBLE_DEVICES={gpu_id+4}\n\n')
        # Run commands sequentially
        for cmd in batch:
            f.write(f'{cmd}\n')
            
    # Make the script executable
    os.chmod(script_name, 0o755)

    session_name = f"gpu_{gpu_id}"

    # Start tmux session that runs the script
    tmux_command = ["tmux", "new-session", "-d", "-s", session_name, f"bash {script_name}"]
    subprocess.run(tmux_command)

    print(f"Started tmux session {session_name} running {len(batch)} commands sequentially on GPU {gpu_id+4}")
