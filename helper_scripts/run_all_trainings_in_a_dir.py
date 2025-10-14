import os
import time
import subprocess

# Directory path containing the files
directory = "/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/hanadi_to_guangzhou/scripts"

# Command template (assuming you want to use the file as an argument to the command)
command_template = "sbatch {file_path}"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Ensure it’s a file, not a directory
    if os.path.isfile(file_path):
        # Construct the command by replacing {file_path} with the actual path
        command = command_template.format(file_path=file_path)
        print(f"Running: {command}")

        # Execute the command
        subprocess.run(command, shell=True)

        # Delay for 40 seconds before moving to the next file
        time.sleep(30)
