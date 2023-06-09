#! /bin/bash

set -e

# Set this to the directory containing empty overlay images
# Note: on GCP the overlay directory does not exist
OVERLAY_DIRECTORY=/scratch/$USER/singularity_files/

IMAGE_DIRECTORY=/scratch/$USER/singularity_files/
# IMAGE_DIRECTORY=/scratch/wz2247/singularity/images/

# Set this to the overlay to use for base packages
BASE_PACKAGES_OVERLAY=overlay-15GB-500K.ext3


# We first extract a pre-defined overlay file
# This first overlay file will contain the base
# packages that already exist in the container
# but in a location we can modify
echo "Extracting base package overlay"
cp $OVERLAY_DIRECTORY/$BASE_PACKAGES_OVERLAY.gz .
gunzip $BASE_PACKAGES_OVERLAY.gz
mv $BASE_PACKAGES_OVERLAY overlay-base.ext3

export SINGULARITY_CACHEDIR=/tmp/$USER
cp -rp /scratch/$USER/Dataset_Student.sqsh /tmp/


# We execute the required commands to obtain a minimal
# new conda environment at the location /ext3/conda/bootcamp
#
# Note that we deactivate binding of the $HOME directory,
# as `conda init` modifies files in the $HOME directory,
# and we want these modifications to be saved in the overlay
# rather than on the filesystem of whichever this script is running on.
#
# However, we still need to bind something in $HOME simply
# to make the directory appear, here we choose to bind $HOME/.ssh
#
echo "Cloning base packages into overlay"
singularity exec --containall --no-home --bind $HOME/.ssh -B /scratch/$USER/DL_project_2023/requirements.txt:/requirements.txt\
    --overlay /tmp/Dataset_Student.sqsh \
    --overlay overlay-base.ext3 \
    $IMAGE_DIRECTORY/pytorch_22.08-py3.sif /bin/bash << 'EOF'
conda create --prefix /ext3/conda/dl-project --clone base
conda init bash
source ~/.bashrc
conda activate /ext3/conda/dl-project
export TMPDIR=/ext3/tmp
mkdir -p $TMPDIR
pwd
pip install -r /requirements.txt
rm -rf $TMPDIR
EOF