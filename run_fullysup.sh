python3 train_fully_supervised.py --learning_rate 10e-2 --wd 2e-3 --moment 0.9 --scheduler True --batch_size 70 --n_epochs 40 --num_classes 2 --model FCN --rotate False --scale False --size_img 150 --size_crop 120 --gpu 2 --split False --split_ratio 0.3 --nw 4 --target_size 256 --stride 128 --entire_image True --save_best True
