import sys
import subprocess

# Path to the Python interpreter
python_executable = sys.executable

# Path to the Python script you want to run
script_path = "train_resnet.py"

for model_name in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
    print("---------- train model using " + model_name + " ------------")
    # Arguments to pass to the Python script
    script_arguments = [model_name]

    # Construct the command to run the script with arguments
    command = [python_executable, script_path] + script_arguments

    # Run the script using subprocess
    subprocess.run(command)
    print("---------- Done (" + model_name + ") ------------")
