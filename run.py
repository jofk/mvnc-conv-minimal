import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose print
import caffe
from mvnc import mvncapi

caffe.set_mode_cpu()

net = caffe.Net("./model/det1_conv1.prototxt", "./model/det1_conv1.caffemodel", caffe.TEST)

input = np.load("input.npy")


net.blobs['data'].data[...] = input
output_caffe = net.forward(end='conv1')
np.savetxt('./output/output_expected.txt', output_caffe['conv1'].flatten())




# Set the global logging level to debug (full verbosity)
mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, mvncapi.LogLevel.DEBUG)

# Get a list of available device identifiers
device_list = mvncapi.enumerate_devices()
if (len(device_list) == 0):
    print("No device found")

# Initialize a Device
device = mvncapi.Device(device_list[0])

# Initialize the device and open communication
device.open()

# Load graph file data
graph_file_path = './model/det1_conv1.graph'
with open(graph_file_path, mode='rb') as f:
    graph_file_buffer = f.read()

# Initialize a Graph object
graph = mvncapi.Graph('graph1')

# Allocate the graph to the device and create input and output Fifos with default arguments
input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_file_buffer)

# Write the tensor to the input_fifo and queue an inference
graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, input, 'user object')

# Get the results from the output queue
output_mvnc, user_obj = output_fifo.read_elem()

np.savetxt('./output/output_mvnc.txt', output_mvnc.flatten())

# Clean up
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()