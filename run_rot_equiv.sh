python3 train_rot_equiv.py --learning_rate 10e-2 --wd 2e-3 --moment 0.9 --scheduler True --batch_size 80 --n_epochs 40 --num_classes 2 --model FCN --rotate False --scale False --gamma 0.9 --multi_task False --size_img 150 --size_crop 120 --gpu 0 --split True --split_ratio 0.3 --nw 4 --target_size 120 --stride 60 --entire_image False --save_best True
