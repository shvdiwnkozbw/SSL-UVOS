python -W ignore src/eval_oneshot.py \
        --basepath /mnt/data/mmlab_ie/qianrui/davis/DAVIS \
        --batch_size 1 \
        --num_t 1 \
        --output_path test_log \
	--dataset DAVIS2017 \
        --resolution 192 384 \
        --ratio 10 \
        --tau 1.0 \
        --save_path DAVIS_Attn \
        --resume_path checkpoint_dino-s-8.pth

rm davis2017-evaluation/results/unsupervised/rvos/*.csv
python davis2017-evaluation/evaluation_method.py --task unsupervised --results_path davis2017-evaluation/results/unsupervised/rvos
