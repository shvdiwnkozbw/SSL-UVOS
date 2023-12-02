python -W ignore src/eval_oneshot.py \
        --basepath /path/to/evaluation/data \
        --batch_size 1 \
        --num_t 1 \
        --output_path test_log \
<<<<<<< HEAD
		--dataset DAVIS2017 \
        --resolution -1 -1 \
=======
	--dataset DAVIS2017 \
        --resolution 192 384 \
>>>>>>> 6de197fcaf7a89347d5bde1f7a5415786dfaef89
        --ratio 10 \
        --tau 1.0 \
        --save_path DAVIS_Attn \
        --resume_path checkpoint_dino-s-8.pth

rm davis2017-evaluation/results/unsupervised/rvos/*.csv
python davis2017-evaluation/evaluation_method.py --task unsupervised --results_path davis2017-evaluation/results/unsupervised/rvos
