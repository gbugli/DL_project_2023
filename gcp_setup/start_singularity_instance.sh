set -e

# The HUGGINGFACE_HUB_CACHE variable defines the directory containing
# downloaded datasets and models from the huggingface hub.
# In order to ensure that there is enough storage, we locate
# that cache in the /scratch directory.
# Inside the container, we mount the standard location for the hub cache
# ~/.cache/huggingface/hub to this location
# HF_HOME=${HF_HOME:-/scratch/$USER/cache/huggingface/}
# mkdir -p $HF_HOME


IMAGE=${IMAGE:-/scratch/$USER/singularity_files/pytorch_22.08-py3.sif}

INSTANCE_NAME=${INSTANCE_NAME:-dl_project}


# Set temporary directory in case it is not set
TMPDIR=${TMPDIR:-/tmp}

# This is the name of the temporary writable overlay to use. 
# By default, it is defined as overlay-temp.ext3, but you may
# re-define it to some other value by setting an environment variable
# before calling this script.
TMP_OVERLAY=${TMP_OVERLAY:-overlay-temp.ext3}
TMP_OVERLAY_SOURCE=${TMP_OVERLAY_SOURCE:-overlay-5GB-3.2M.ext3}


# Set this to the directory containing empty overlay images
# Note: on GCP the overlay directory does not exist
OVERLAY_DIRECTORY=/scratch/$USER/singularity_files/

# First, check that the temp overlay exists. Otherwise grap it from the overlays.

if [[ ! -f $TMP_OVERLAY ]]; then

echo "Temporary overlay not found, automatically creating a new one."
cp "$OVERLAY_DIRECTORY/$TMP_OVERLAY_SOURCE.gz" "$TMPDIR"
gunzip "$TMPDIR/$TMP_OVERLAY_SOURCE.gz"
mv "$TMPDIR/$TMP_OVERLAY_SOURCE" "$TMP_OVERLAY"

fi


# This script starts singularity with all the expected binds in place.
# The following binds / overlays are defined

# --containall --no-home Ensures that the container is well isolated
# -B $HOME/.ssh: binds the ssh directory to ensure that ssh authorized keys are propagated
# -B /scratch: binds the entire /scratch filesystem
# -B $PWD: binds the current working directory
# --nv: enables CUDA integration for the container to pass-through GPUs
# --overlay overlay-base.ext3: overlay with the base packages, created by scripts/create_base_overlay.sh
# --overlay overlay-packages.ext3: overlay with our installed packages, created by scripts/create_package_overlay.sh

singularity instance start --containall --no-home -B $HOME/.ssh -B /scratch -B $PWD --nv \
    --overlay overlay-temp.ext3 \
    --overlay overlay-base.ext3:ro \
    --overlay /scratch/$USER/Dataset_Student.sqsh:ro \
    $IMAGE ${INSTANCE_NAME}