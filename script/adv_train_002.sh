export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
unset CUDA_VISIBLE_DEVICES

nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 022 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 023 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 030 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 031 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 032 &

echo quit | nvidia-cuda-mps-control

