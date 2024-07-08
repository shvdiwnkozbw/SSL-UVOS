python -W ignore -m torch.distributed.launch --nproc_per_node=8 --use_env \
		src/train_RGB_cluster.py \
        --basepath /path/to/data \
        --batch_size 1 \
        --seed 0 \
        --grad_iter 0 \
        --num_t 1 \
        --lr 1e-5 \
        --num_frames 3 \
        --output_path test_log \
        --dino_path dino_small_8.pth \
		--dataset DAVIS