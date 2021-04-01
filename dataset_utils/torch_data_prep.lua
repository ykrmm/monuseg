require 'io'
require 'torch'
require 'image'

------------------------------ Parameters ---------------------------------
file_name = '/home/sanuj/Projects/nuclei-net-data/train.txt'
save_name = '/home/sanuj/Projects/nuclei-net-data/train.t7'

num_images = 10000*3
num_channels = 3
width = 51
height = 51
---------------------------------------------------------------------------

file = io.open(file_name, 'rb')
data = torch.Tensor(num_images, num_channels, width, height):byte()
label = torch.Tensor(num_images):byte()
counter = 1

for line in file:lines() do
	print(counter)
	image_name, image_label = line:split(' ')[1], line:split(' ')[2]
	data[counter] = image.load(image_name, num_channels, 'byte')
	label[counter] = image_label
	counter = counter + 1
end

torch.save(save_name, {data = data, label = label})