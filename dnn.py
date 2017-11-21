#coding=UTF-8
'''
DNN的简单实现,隐藏层使用relu函数， 输出层使用sigmoid函数
'''
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist =  input_data.read_data_sets('MNIST_data', one_hot = True)


class DNN:

    def __init__(self, network_layer, learning_rate, loss_func='sigmoid'):
        self.network_layer = network_layer
        self.layers = len(network_layer)
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.weights, self.biases = self.get_weights_biases(self.network_layer)
        
    def relu(self,arr):
        return np.where(arr>0, arr, 0)

    def deri_relu(self,arr):
        return np.where(arr>0,1,0)

    def sigmoid(self,arr):
        return 1/(1+np.exp(-arr))

    def deri_sigmoid(self,arr):
        return arr * (1 - arr)
    
    def softmax(self,arr):
#         print(arr[0])   overflow and underflow problem not solved
        arr -= np.max(arr, axis=1).reshape([-1,1])
        s = np.sum(np.exp(arr), axis=1).reshape([-1,1])
        return np.exp(arr)/s
    
    def get_weights_biases(self, network_layer):
        weights = []
        biases = []
        for i in range(self.layers-1):
            weights.append(1/np.sqrt(network_layer[i])*np.random.randn(network_layer[i], network_layer[i+1]))
            biases.append(np.random.random((1, network_layer[i+1])))
        return weights, biases

    def inference(self, input_features):
        h_out = input_features
        h = []
        h.append(h_out)
        for i in range(self.layers-2):
            h_out = self.relu(np.dot(h_out,self.weights[i]) + self.biases[i])
            h.append(h_out)
        if self.loss_func == 'sigmoid':
            logits = self.sigmoid(np.dot(h[-1], self.weights[-1]) + self.biases[-1])
        elif self.loss_func == 'softmax':
            logits = self.softmax(np.dot(h[-1], self.weights[-1]) + self.biases[-1])
        h.append(logits)
        return h

    def back_pro_last_layer(self,h, output_labels):
        if self.loss_func == 'sigmoid':
            self.err = (h[-1] - output_labels) * self.deri_sigmoid(h[-1])/output_labels.shape[0]
        elif self.loss_func == 'softmax':
            self.err = (h[-1] - output_labels)/output_labels.shape[0]
        self.weights[-1] -= self.learning_rate * np.dot(h[-2].T, self.err)
        self.biases[-1] -= self.learning_rate * np.reshape(np.sum(self.err,axis=0), (1,-1))

    def back_pro_pre_layer(self,h):
        for i in range(self.layers-2):
            self.err = np.dot(self.err, self.weights[self.layers-i-2].T) * self.deri_relu(h[self.layers-i-2])
            self.weights[self.layers-i-3] -= self.learning_rate * np.dot(h[self.layers-i-3].T, self.err)
            self.biases[self.layers-i-3] -= self.learning_rate * np.reshape(np.sum(self.err,axis=0), (1,-1))

    def back_pro(self, h, output_labels):
        self.back_pro_last_layer(h, output_labels)
        self.back_pro_pre_layer(h)

        
    def get_loss(self, logits, labels):
        
        if self.loss_func == 'sigmoid':
            loss = np.mean((logits-labels)**2)
        elif self.loss_func == 'softmax':
            loss = -np.mean(labels*np.log(logits))
        return loss
    
    def get_accuracy(self, inputs, labels):
        logits = self.inference(inputs)[-1]
        acc = np.mean((np.equal(np.argmax(logits,axis=1), np.argmax(labels,axis=1))))    
        return acc
        
    def train_network(self, input_x, label_y):
        logits = my_dnn.inference(input_x)
        my_dnn.back_pro(logits, label_y)

    def predict(self, input_feature):
        h = self.inference(input_feature)
        return np.argmax(h[-1], axis=1)



batch_size = 64
epochs = 10
display_step = 200
network_layer = [784,256,256,10]
my_dnn = DNN(network_layer, 0.01, loss_func='softmax')
for epoch_i in range(epochs):
    total_batch = int(mnist.train._num_examples/batch_size)
    for batch_i in range(total_batch):
        input_x, label_y = mnist.train.next_batch(batch_size)
        my_dnn.train_network(input_x, label_y)
        if batch_i % display_step == 0:
            print('eopch:{}/{}, batch:{}/{}, acc:{}'.format(epoch_i,epochs,batch_i, total_batch,my_dnn.get_accuracy(input_x, label_y)))



