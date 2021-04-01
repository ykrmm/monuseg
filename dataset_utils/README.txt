data
----

create_maps.m : this is used to create ternary maps from annotations and original image.

Data is prepared by first generating it for Caffe and then converting it to torch. This is done because Caffe data requires a text file with mapping from image name to labels. It is easier to shuffle and concatenate text files to control the number of samples/class in training data.

You need to change the following parameters in caffe_data_prep.py:
- root
- path
- file_name
- mask_name

Call "python caffe_data_prep.py"

This will generate multiple folders and text files. Each folder has around 25000 51x51 size patches with it's meta.txt file. You can concatenate and shuffle these files using "cat" and "shuf" from terminal.

Eg: Suppose 3 folders are generated: 1, 2, 3
To concatenate: "cat 1/meta.txt 2/meta.txt 3/meta.txt > train.txt"
To shuffle: "shuf --output=train.txt train.txt"

You can use "count_labels.py" and "remove_labels.py" to count number of samples for each label and remove number of samples for each label after you have "train.txt" and "test.txt".

Once you have "train.txt" and "test.txt" then call "torch_data_prep.lua" to convert this data to torch format.

-------------------------------------------------------------------