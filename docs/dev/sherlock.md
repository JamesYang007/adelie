# Sherlock Tips

Oftentimes, we must run jobs on clusters (such as Sherlock at Stanford University)
that fit group lasso on large-scale datasets.
This is a personal set of notes to keep in mind for setting up `adelie` on Sherlock.
On different clusters, these notes may not apply.

## Module Setup

Before installing `adelie` on Sherlock, we must load the following modules:
```
ml python/3.9
ml gcc/9
```

## Installation

First, clone the repository.
Then, run the following script to install on Sherlock:
```bash
#!/bin/bash
#SBATCH --job-name=install_adelie
#SBATCH --output=install_adelie.%j.out
#SBATCH --error=install_adelie.%j.err
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -p <partition>

module load python/3.9
module load gcc/9
cd <adelie path>
pip3 install -e .
```
where `<partition>` is your favorite partition on the cluster and `<adelie path>` is the local path to the cloned repository.
__Note: if RAM is too small (e.g. default setting), then the installation fails!__
This will install `adelie` in editable mode, which is the preferred mode for development.