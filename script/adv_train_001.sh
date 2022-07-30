export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
unset CUDA_VISIBLE_DEVICES

nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 000 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 001 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 002 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 003 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 010 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 011 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 012 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 013 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 020 &
nohup python3 ../src/adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch 021 &

echo quit | nvidia-cuda-mps-control

