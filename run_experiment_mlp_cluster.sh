#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 2	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=40000  # memory in Mb
#SBATCH --time=02-08:00:00


# export CUDA_HOME=opt/cuda-11.5.0/

# export CUDNN_HOME=/opt/cudnn-11.4-8.2.2.26/

# export STUDENT_ID=$(whoami)

# export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

# export CPATH=${CUDNN_HOME}/include:$CPATH

# export PATH=${CUDA_HOME}/bin:${PATH}

# export PYTHON_PATH=$PATH

# mkdir -p /disk/scratch/${STUDENT_ID}


SCRATCH_DISK=/disk/scratch
dest_path=${SCRATCH_DISK}/${USER}/yolov7_carla_object_detection/carla
mkdir -p ${dest_path}
src_path=carla.tar.xz
rsync --archive --update --compress --progress ${src_path} ${dest_path}
tar -xvf ${dest_path}/carla.tar.xz


# Activate the relevant virtual environment:


source /lustre/${STUDENT_ID}/miniconda3/bin/activate mlp

python -m torch.distributed.launch \
                                    --nproc_per_node 2 \
                                    --master_port 9527 \
                                    train.py \
                                    --workers 4 \
                                    --device 0,1 \
                                    --sync-bn \
                                    --batch-size 156 \
                                    --data data/carla_mlp_cluster.yaml \
                                    --img 608 608 \
                                    --cfg cfg/training/yolov7-carla.yaml \
                                    --weights '' \
                                    --name yolov7-carla-ob-det \
                                    --hyp data/hyp.scratch.p5.yaml

