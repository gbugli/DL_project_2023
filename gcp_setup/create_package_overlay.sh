#! /bin/bash

set -e

# Set this to the directory containing empty overlay images
# Note: on GCP the overlay directory does not exist
OVERLAY_DIRECTORY=/scratch/$USER/singularity_files/

IMAGE_DIRECTORY=/scratch/$USER/singularity_files/
# IMAGE_DIRECTORY=/scratch/wz2247/singularity/images/

# Set this to the overlay to use for additional packages.
ADDITIONAL_PACKAGES_OVERLAY=overlay-15GB-500K.ext3

# We will install our own packages in an additional overlay
# So that we can easily reinstall packages as needed without
# having to clone the base environment again.
echo "Extracting additional package overlay"
cp $OVERLAY_DIRECTORY/$ADDITIONAL_PACKAGES_OVERLAY.gz .
gunzip $ADDITIONAL_PACKAGES_OVERLAY.gz
mv $ADDITIONAL_PACKAGES_OVERLAY overlay-packages.ext3


# We now execute the commands to install the packages that we need.
echo "Installing additional packages"
singularity exec --containall --no-home -B $HOME/.ssh -B /scratch/$USER/NYU-Zillow-Capstone-2022-Team-D/requirements.txt:/requirements.txt\
    --overlay overlay-packages.ext3 \
    --overlay overlay-base.ext3:ro \
    $IMAGE_DIRECTORY/pytorch_22.08-py3.sif /bin/bash << 'EOF'
source ~/.bashrc
conda activate /ext3/conda/cds-capstone
export TMPDIR=/ext3/tmp
mkdir -p $TMPDIR
pwd
ls
pip install -r /requirements.txt
rm -rf $TMPDIR
EOF