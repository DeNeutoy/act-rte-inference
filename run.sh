#!/bin/bash -l

echo 'Running Model'


#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=24:00:00
#These are optional flags but you problably want them in all jobs

#$ -S /bin/bash
#$ -N test
#$ -wd /home/mneumann/act-rte-inference/

export PYTHONPATH=${PYTHONPATH}:/home/mneumann/act-rte-inference/
LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64:/share/apps/gcc-5.2.0/lib:/opt/gridengine/lib/linux-x64:/opt/gridengine/lib/linux-x64:/opt/openmpi/lib:/opt/python/lib:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64' \
                ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
                ./src/train.py \
                --model_size "small" \
                --data "./snli_1.0" \
                --weights_dir "./out/test_run" \
                --verbose True \
                --debug False \
