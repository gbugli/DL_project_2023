scp -r $USER@greene-dtn.hpc.nyu.edu:/scratch/gb2572/Dataset_Student.tar.gz /scratch/$USER/

tar -xvzf Dataset_Student.tar.gz

mkdir /scratch/$USER/singularity_files/

scp $USER@greene-dtn.hpc.nyu.edu:/scratch/wz2247/singularity/images/pytorch_22.08-py3.sif /scratch/$USER/singularity_files/

scp $USER@greene-dtn.hpc.nyu.edu:/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/singularity_files/

# scp $USER@greene-dtn.hpc.nyu.edu:/scratch/work/public/overlay-fs-ext3/overlay-5GB-3.2M.ext3.gz /scratch/$USER/singularity_files/