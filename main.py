import tensorflow as tf
from tensorflow import keras

from PIL import Image
import random

sample_size = 8


def get_image(path):
    im = Image.open(path, 'r').convert('RGB')
    width, height = im.size
    data = im.load()
    output = []
    for y in range(0, height):
        output.append([])
        for x in range(0, width):
            output[y].append([float(i) for i in data[x, y]])
    return output, width, height

def save_image(path, data, width, height):
    im = Image.new('RGB', (width, height))
    pixels = im.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = tuple([min(max(int(i), 0), 256) for i in data[y][x]])
    im.save(path)

def get_sub(data, width, height, xoff, yoff, size):
    output = []
    for y in range(yoff, yoff + size):
        row = []
        for x in range(xoff, xoff + size):
            row.append(data[y][x])
        output.append(row)
    return output

def make_samples(data, width, height, sample_size):
    output = []
    for yoff in range(0, height, sample_size):
        for xoff in range(0, width, sample_size):
            output.append(get_sub(data, width, height, xoff, yoff, sample_size))
    return output

def stitch_samples(data, width, height, sample_size):
    output = []
    x_samples = int(width/sample_size)
    y_samples = int(height/sample_size)
    print(x_samples, y_samples)
    for y in range(height):
        output.append([])
        for x in range(width):
            #print(int(x/x_samples), int(y/y_samples))
            output[y].append(data[int(x/sample_size) + int(y/sample_size) * x_samples][y%sample_size][x%sample_size])
    return output


#save_image('test.png', data, width, height)
input_samples = make_samples(*get_image('low_octahedron.png'), sample_size)
output_samples = make_samples(*get_image('octahedron.png'), sample_size)
samples = [item for item in zip(input_samples, output_samples)]

test = stitch_samples(input_samples, 1024, 1024, sample_size)

save_image('stest.png', test, 1024, 1024)
save_image('in.png', samples[0][0], sample_size, sample_size)
save_image('out.png', samples[0][1], sample_size, sample_size)

# setup the model
image_input = tf.placeholder(tf.float32, [None, sample_size, sample_size, 3])
image_output = tf.placeholder(tf.float32, [None, sample_size, sample_size, 3])
learning_rate = tf.placeholder(tf.float32, None)

filter_shape = [sample_size, sample_size, 3, 3]
filter1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=1.0))
conv1 = tf.nn.conv2d(image_input, filter1, [1, 1, 1, 1], 'SAME')

filter2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5))
conv2 = tf.nn.conv2d(conv1, filter2, [1, 1, 1, 1], 'SAME')

loss = tf.reduce_sum(tf.square(conv2 - image_output))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    rate = 0.95
    for i in range(25000):
        train = random.sample(samples, 100)
        train_in = [i[0] for i in train]
        train_out = [i[1] for i in train]

        rate *= 0.9995
        _, error = session.run([minimize, loss], {image_input: train_in, image_output: train_out, learning_rate: rate})
        print('epoch {}: {}'.format(i, error))
        if i % 250 == 0:
            test_data = session.run(conv2, {image_input: [input_samples[0]]})
            save_image('test.png', test_data[0], sample_size, sample_size)
        if i % 1000 == 0:
            test_data = session.run(conv2, {image_input: input_samples})
            out_data = stitch_samples(test_data, 1024, 1024, sample_size)
            save_image('full_test.png', out_data, 1024, 1024)
    test_data = session.run(conv2, {image_input: [input_samples[0]]})
    print(test_data)
    save_image('test.png', test_data[0], sample_size, sample_size)
    test_data = session.run(conv2, {image_input: input_samples})
    out_data = stitch_samples(test_data, 1024, 1024, sample_size)
    save_image('full_test.png', out_data, 1024, 1024)

