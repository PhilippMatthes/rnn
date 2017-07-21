from __future__ import print_function, division
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.rnn as rnn
from probprint import probprint

corpora = []
from nltk.corpus import brown as corpus1
from nltk.corpus import gutenberg as corpus2
from nltk.corpus import nps_chat as corpus3
corpora.append(corpus1)
corpora.append(corpus2)
corpora.append(corpus3)

import os
def clear():
    os.system('cls' if os.name=='nt' else 'clear')

data = ""
for corpus in corpora:
    data += " ".join(corpus.words())

# with open("input/input.txt", "r") as f:
#     data = f.read().replace("\n","")
chars = list(sorted(set(data)))

VOCAB_SIZE = len(chars)
INPUT_SIZE = len(data)

print("Input size: "+str(INPUT_SIZE))
print("Vocab size: "+str(VOCAB_SIZE))
print("Characters: "+str(chars))

ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}


total_series_length = 5000000
num_epochs = int(INPUT_SIZE/total_series_length)*10 - 1
truncated_backprop_length = 15
state_size = 100
num_classes = VOCAB_SIZE
echo_step = 1
batch_size = 40
num_layers = 3
num_batches = total_series_length//batch_size//truncated_backprop_length


def generateData(current_chunk):
    if current_chunk == 0:
        current_read_position = (int)(total_series_length/10)
    else:
        current_read_position = (int)(current_chunk*total_series_length/10)
    x_chars = data[current_read_position:current_read_position+total_series_length]
    x = np.asarray([char_to_ix[value] for value in x_chars])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)


W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)

# Forward passes
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
states_series, current_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(batchX_placeholder, -1), initial_state=rnn_tuple_state)
states_series = tf.reshape(states_series, [-1, state_size])

logits = tf.matmul(states_series, W2) + b2 #Broadcasted addition
labels = tf.reshape(batchY_placeholder, [-1])

logits_series = tf.unstack(tf.reshape(logits, [batch_size, truncated_backprop_length, num_classes]), axis=1)
predictions_series = [tf.nn.softmax(logit) for logit in logits_series]


losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
total_loss = tf.reduce_mean(losses)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.5
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.99, staircase=True)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss,global_step=global_step)

def plot(loss_list):
    # plt.subplot(2, 3, 1)
    axes = plt.gca()
    length = len(loss_list)
    axes.set_xlim([length-10000,length])
    plt.cla()
    plt.plot(loss_list,alpha=0.5)
    # plt.plot(median_list,alpha=0.5)
    # plt.plot(learning_rate)
    plt.yscale('log')
    plt.draw()
    plt.pause(0.00000001)


with tf.Session() as sess:
    saver = tf.train.Saver()

    reset = input("Reset network? (y/n)\n") == "y"
    if reset:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, "state/model.ckpt")
        print("Model restored.")

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    median_list = []
    learning_rate_list = []

    try:
        for epoch_idx in range(num_epochs):
            x,y = generateData(epoch_idx)
            _current_state = np.zeros((num_layers, 2, batch_size, state_size))

            print("New data, epoch", epoch_idx)

            for batch_idx in tqdm(range(num_batches)):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:,start_idx:end_idx]
                batchY = y[:,start_idx:end_idx]

                _total_loss, _train_step, _current_state, _predictions_series, _learning_rate, _logits, _labels, _states_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series, learning_rate, logits, labels, states_series],
                    feed_dict={
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        init_state: _current_state
                    })

                loss_list.append(_total_loss)
                learning_rate_list.append(_learning_rate)

                if batch_idx%100 == 0:
                    clear()
                    prediction = []
                    for neuron_output_list in _predictions_series:
                        output = neuron_output_list[0]
                        scalar = 0.5
                        for neuron_output in neuron_output_list[1:]:
                            output = np.add(output, neuron_output * scalar)
                            scalar = scalar / 2
                        output_list = output.tolist()
                        prediction.append((output_list.index(max(output_list)), max(output_list)))
                    print("Step",batch_idx, "Loss", _total_loss,"\n")
                    print("Epoch",epoch_idx,"out of",num_epochs,"\n")
                    # print("\nX:\n"+"".join([ix_to_char[value] for value in batchX[0]]))
                    print("\ny:\n"+"".join([ix_to_char[value] for value in batchY[0]]))
                    maxvalue = 1
                    for value in prediction:
                        if (value[1] > maxvalue):
                            maxvalue = value[1]
                    print("\nPrediction:\n"+"".join([probprint(ix_to_char[value[0]], value[1], maxvalue) for value in prediction])+"\n")
                    plt.ioff()
                    plot(loss_list)

    except KeyboardInterrupt:
        pass

    save_path = saver.save(sess, "state/model.ckpt")
    print("Model saved in file: %s" % save_path)

plt.ioff()
