echo '--- Now inside container ---'
pip install --upgrade transformers==4.57.1 accelerate
pip install trl==0.8.6
pip install bitsandbytes==0.48.1
pip install deepspeed==0.16.9
cd /home/users/nus/e1352533/code/sft/LLaMA-Factory && CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/train.py /home/users/nus/e1352533/code/sft/holmes_full_sft.yaml
