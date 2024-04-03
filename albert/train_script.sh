#!/bin/bash

#SBATCH --job-name=albert-train
#SBATCH --mail-type=BEGIN,END

# Change this to email
#SBATCH --mail-user=e0735378@u.nus.edu 

# Change this for other partition
#SBATCH --partition=medium

# Change this if need other time limit
#SBATCH --time=60

## Just useful logfile names

#SBATCH --output=logs/albert-train-output_%j.slurmlog
#SBATCH --error=logs/albert-train-error_%j.slurmlog

echo "Job is running on $(hostname), started at $(date)"

# Get some output about GPU status before starting the job
nvidia-smi 

# Export path for installation dependencies
# CHANGE the path to :/home/f/{username}/.loca/bin
export PATH="$PATH:/home/f/farrelds/.local/bin"

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
echo "Running!"
srun python pipeline.py

# Training completed
echo -e "\n====> Finished running.\n"
echo -e "\nJob completed at $(date)"