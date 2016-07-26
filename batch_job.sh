#!/bin/bash -l

echo 'Running Model'
#$ -l tmem=12G
#$ -l h_vmem=12G
#$ -l h_rt=30:00:00
#These are optional flags but you problably want them in all jobs

#$ -S /bin/bash
#$ -N adaptive-IAA-gridsearch
#$ -wd /home/neumann/act-rte-inference
#$ -t 1-2
#$ -o ./out/grid_outputs/

export PYTHONPATH=${HOME}/.local/lib/python3.4/site-packages:${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:/home/mneumann/
timestamp=`date -u +%Y-%m-%dT%H%MZ`
mkdir "./out/grid_outputs/${timestamp}"
mkdir "./out/grid_outputs/${timestamp}/grid_output"


i=$(expr $SGE_TASK_ID - 1)

learning_rate=(0.001 0.0001)
hidden_size=(128 256)
keep_prob=(0.8 0.6)
eps=(0.01 0.2)
step_penalty=(0.01 0.001 0.0001 0.00001)

total_steps=$((${#learning_rate[@]} * ${#hidden_size[@]} * ${#keep_prob[@]} * ${#eps[@]} * ${#step_penalty[@]}))
echo ${total_steps}

i=$((i % total_steps))
steps=$(($total_steps / ${#learning_rate[@]}))
echo ${total_steps}
learning_rate_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#hidden_size[@]}))
echo ${steps}
hidden_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#keep_prob[@]}))
echo ${steps}
keep_prob_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#eps[@]}))
eps_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#step_penalty[@]}))
echo ${steps}
step_penalty_idx=$((i / steps))

name="AIAA_lr_${learning_rate[learning_rate_idx]}hid_${hidden_size[hidden_size_idx]}drop_${keep_prob[keep_prob_idx]}eps_${eps[eps_idx]}step_pen_${step_penalty[step_penalty_idx]}"
echo ${name}
LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64:/share/apps/gcc-5.2.0/lib:/opt/gridengine/lib/linux-x64:/opt/gridengine/lib/linux-x64:/opt/openmpi/lib:/opt/python/lib:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64' \
  ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
                ./src/train_extra.py \
                --model "AdaptiveIAAModel" \
                --data "./snli_1.0" \
                --weights_dir "./out/grid_outputs/${timestamp}/grid_output" \
                --vocab_path "./snli_1.0/unbounded_vocab.txt" \
                --embedding_path "../glove/glove.6B.300d.txt" \
                --verbose True \
                --grid_search \
                --learning_rate ${learning_rate[learning_rate_idx]} \
                --hidden_size ${hidden_size[hidden_size_idx]} \
                --keep_prob ${keep_prob[keep_prob]} \
                --eps ${eps[eps_idx]} \
                --step_penalty ${step_penalty[step_penalty_idx]} | tee "./out/grid_outputs/${timestamp}/${name}.txt"