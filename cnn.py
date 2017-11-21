'''
input_img shape is [batch_size, in_channels, img_row, img_col]
weight shape is [in_channels, out_channels, kernel_row, kernel_col]
'''

import math
import numpy as np
import time

def padd_operation(input_x, w_shape, stride, padding, func='conv'):    
    
    B,C,H,W = input_x.shape
    if padding=='VALID':

        h_idx = int(math.floor((H-w_shape[0]))/stride[0]) + 1
        v_idx = int(math.floor((W-w_shape[1]))/stride[1]) + 1
        drop_row = (H-w_shape[0])%stride[0]
        drop_col = (W-w_shape[1])%stride[1]
        output_pad = input_x[:,:,:H-drop_row,:W-drop_col]

    elif padding=='SAME':

        h_idx = math.ceil(H/stride[0])
        v_idx = math.ceil(W/stride[1])
        pad_size = [w_shape[0]+(h_idx-1)*stride[0]-H, w_shape[1]+(v_idx-1)*stride[1]-W]
        pad_right = math.ceil(pad_size[1]/2)
        pad_left = pad_size[1] - pad_right
        pad_down = math.ceil(pad_size[0]/2)
        pad_up = pad_size[0] - pad_down
        pad_value = -np.float32('inf') if func == 'pool' else 0
        output_pad = np.pad(input_x, ((0,0),(0,0),(pad_up,pad_down),(pad_left, pad_right)),\
                              'constant', constant_values=pad_value)
    return h_idx, v_idx, output_pad

def img2col(input_x, ksize, stride, h_idx, v_idx):

    ker_row, ker_col = ksize
    i_fro = [range(i*stride[0],i*stride[0]+ker_row) for i in range(h_idx)]
    j_fro = [range(i*stride[1],i*stride[1]+ker_col) for i in range(v_idx)]
    i = np.repeat(np.repeat(i_fro, ker_col, axis=1), v_idx, axis=0)
    j = np.tile(np.tile(j_fro, ker_row), [h_idx,1])
    x_col = np.transpose(input_x[:,:,i,j], [0,2,1,3]).reshape(input_x.shape[0],h_idx*v_idx,-1)
    return x_col

def col2img(input_x, ksize, stride, h_idx, v_idx):

    ker_row, ker_col = ksize
    i_fro = [range(i*h_idx,(i+1)*h_idx) for i in range(h_idx)]
    j_fro = [range(i*ker_col,(i+1)*ker_col) for i in range(ker_row)]
    i = np.repeat(np.repeat(i_fro, ker_col, axis=1), ker_row, axis=0)
    j = np.tile(np.tile(j_fro, v_idx), [h_idx,1])
    return input_x[:,:,i,j]

def reverse_padd(input_x, raw_row, raw_col, padding):

    if padding == 'SAME':
        down = math.ceil((input_x.shape[2] - raw_row)/2)
        up = input_x.shape[2] - raw_row - down
        right = math.ceil((input_x.shape[3] - raw_col)/2)
        left = input_x.shape[3] - raw_col - right
        output = np.delete(input_x, list(range(0,up))+list(range(input_x.shape[2]-down, input_x.shape[2])), axis=2)
        output = np.delete(output, list(range(0,left))+list(range(input_x.shape[3]-right, input_x.shape[3])), axis=3)
    elif padding == 'VALID':
        right = raw_row - input_x.shape[2]
        down = raw_col - input_x.shape[3]
        output = np.pad(input_x, ((0,0),(0,0),(0,down),(0,right)), 'constant')
    return output

class Convlayer:

    def __init__(self, ksize, stride=[1,1], learning_rate=0.01, padding='VALID'):
        
        self.ksize = ksize
        self.in_channel, self.out_channel, self.ker_row, self.ker_col = ksize
        self.weight = np.random.randn(self.in_channel, self.out_channel, self.ker_row, self.ker_col)
        self.bias = np.random.random((self.out_channel))
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
  
    def forward(self, input_x):

        self.batch_size, self.in_channel, self.input_row,  self.input_col = input_x.shape
        self.h_idx,self.v_idx, self.input_tensor = padd_operation(input_x, self.ksize[2:4], self.stride, self.padding, func='conv')
        x_col = img2col(self.input_tensor, self.ksize[2:4], self.stride, self.h_idx, self.v_idx)
        w = np.transpose(self.weight, [0,2,3,1]).reshape(self.in_channel*self.ker_row*self.ker_col,-1)
        output_y = np.dot(x_col, w) + self.bias
        output_y = output_y.reshape(self.batch_size, self.h_idx, self.v_idx, self.out_channel) 
        output_y = output_y.transpose(0,3,1,2)
        return output_y

    def backward(self, err):

        err = err.reshape((self.batch_size, self.out_channel, self.h_idx, self.v_idx))
        expand_shape = [err.shape[2]+(err.shape[2]-1)*(self.stride[0]-1), err.shape[3]+(err.shape[3]-1)*(self.stride[1]-1)]
        expand_err = np.zeros([self.batch_size, self.out_channel, expand_shape[0], expand_shape[1]])
        
        k = np.array([range(0,expand_shape[0],self.stride[0])]).T
        n = np.array([range(0,expand_shape[1], self.stride[1])])
        expand_err[:,:,k,n] = err
        padd_err = self.padd(expand_err)
       
        grad_b = np.sum(expand_err, axis=(0,2,3))
        
        _input = self.input_tensor.transpose([1,0,2,3])
        grad_w_col = img2col(_input, expand_err.shape[2:4], [1,1], self.ker_row, self.ker_col)
        expand_err = np.transpose(expand_err, [0,2,3,1]).reshape(self.batch_size*expand_shape[0]*expand_shape[1], -1)
        grad_w = np.dot(grad_w_col, expand_err)
        grad_w = grad_w.reshape([self.in_channel, self.ksize[2], self.ksize[3], self.out_channel])
        grad_w = grad_w.transpose(0,3,1,2)
        
        rot_weight = self.rot180(self.weight).transpose([1,0,2,3])
        grad_x_col = img2col(padd_err, rot_weight.shape[2:4], [1,1], self.input_tensor.shape[2], self.input_tensor.shape[3])
        rot_weight = np.transpose(rot_weight, [0,2,3,1]).reshape([self.out_channel*self.ker_row* self.ker_col, -1])
        grad_x = np.dot(grad_x_col, rot_weight)
        grad_x = grad_x.reshape([self.batch_size, self.input_tensor.shape[2], self.input_tensor.shape[3], self.in_channel])
        grad_x = grad_x.transpose(0,3,1,2)

        grad_x = reverse_padd(grad_x, self.input_row, self.input_col, self.padding)
        self.weight -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
        return grad_x, [grad_w, grad_b]

    def padd(self, input_x):

        padd_shape = [self.batch_size, self.out_channel, self.ker_row*2+input_x.shape[2]-2, self.ker_col*2+input_x.shape[3]-2]
        padd_err = np.zeros(padd_shape)
        padd_err[:,:,self.ker_row-1:self.ker_row+input_x.shape[2]-1, self.ker_col-1:self.ker_col+input_x.shape[3]-1] = input_x
        return padd_err

    def rot180(self, input_x):

        rot_w = np.zeros_like(input_x)
        k = np.array([range(self.ker_row-1,-1,-1)]).T
        n = np.array([range(self.ker_col-1,-1,-1)])
        rot_w[:,:,k,n] = input_x
        return rot_w

class Maxpoollayer:

    def __init__(self, ksize, stride=[2,2], padding='VALID'):

        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def forward(self, input_x):
        
        self.raw_inrow,self.raw_incol = input_x.shape[2:4]
        self.batch_size, self.in_channel, self.input_row, self.input_col = input_x.shape
        self.h_idx, self.v_idx ,self.input_tensor= padd_operation(input_x, self.ksize, self.stride, self.padding, func='pool')
        output_y = img2col(self.input_tensor, self.ksize, self.stride, self.h_idx, self.v_idx)
        output_y = output_y.reshape(self.batch_size, self.h_idx*self.v_idx, self.in_channel, -1)
        output_y = output_y.transpose([0,2,1,3])
        self.idx = np.argmax(output_y, axis=3)
        output_y = np.max(output_y, axis=3).reshape(self.batch_size, self.in_channel, self.h_idx, self.v_idx)
        return output_y

    def backward(self, err):

        err = err.reshape(self.batch_size, self.in_channel, self.h_idx, self.v_idx)
        grad_x = np.zeros([self.batch_size, self.in_channel, self.h_idx*self.v_idx, self.ksize[0]*self.ksize[1]])
        i = np.repeat(np.tile(np.array([range(self.batch_size)]).T, [1,self.in_channel]), self.h_idx*self.v_idx, axis=1).reshape(self.idx.shape)
        j = np.repeat(np.repeat([range(self.in_channel)], self.batch_size, axis=0), self.h_idx*self.v_idx, axis=1).reshape(self.idx.shape)
        k = np.tile(range(self.h_idx*self.v_idx), self.batch_size*self.in_channel).reshape(self.idx.shape)
        grad_x[i,j,k,self.idx] = err.reshape(self.batch_size, self.in_channel, -1)
        grad_x = col2img(grad_x, self.ksize, self.stride, self.h_idx, self.v_idx)
        grad_x = reverse_padd(grad_x, self.raw_inrow, self.raw_incol, self.padding)
        return grad_x, []



class FClayer:

    def __init__(self, W, learning_rate=0.01):

        self.weight = np.random.randn(W[0],W[1])/np.sqrt(W[0])
        self.bias = np.random.random((1,W[1]))
        self.learning_rate = learning_rate

    def forward(self, input_x):

        self.batch_size = input_x.shape[0]
        self.input_tensor = np.reshape(input_x, [self.batch_size, -1])
        self.output_y = np.dot(self.input_tensor, self.weight) + self.bias
        return self.output_y

    def backward(self, err):

        grad_w = np.dot(self.input_tensor.T, err)
        grad_b = np.sum(err, axis=0).reshape(1,-1)
        err = np.dot(err, self.weight.T)
        self.weight -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
        return err,  [grad_w, grad_b]

class Relulayer:

    def __init__(self):
        pass

    def forward(self, input_x):

        self.output_y =  input_x
        return np.maximum(self.output_y, 0)

    def backward(self, err):

        err = err.reshape(self.output_y.shape)
        err = np.where(self.output_y>0, err, 0)
        return err, []

class Softmaxlayer:

    def __init__(self):
        pass

    def forward(self, input_x):

        input_x -= np.max(input_x, axis=1).reshape([-1,1])
        e_sum = np.sum(np.exp(input_x), axis=1).reshape([-1,1])
        self.output_y = np.exp(input_x)/e_sum
        return self.output_y

    def backward(self, err):

        err = err/self.output_y.shape[0]
        return err,[]

class Sigmoidlayer:

    def __init__(self):
        pass

    def forward(self, input_x):

        self.output_y = 1/(1+np.exp(-input_x))
        return self.output_y

    def backward(self, err):
        
        err = (err * self.output_y * (1-self.output_y))/err.shape[0]
        return err, []

class Net:
    
    def __init__(self):
        
        self.cnn = []
        self.grads = []

    def add(self, layer):

        self.cnn.append(layer)

    def inference(self, input_x):

        logits = input_x.copy()
        for i in self.cnn:
            logits = i.forward(logits)
            # print(i.__class__.__name__,logits.shape)
        return logits

    def update_weigth_bias(self, logits, label_y):

        err = logits - label_y
        for i in reversed(self.cnn):
            err, grad = i.backward(err)
            self.grads.append(grad)
            # print(i.__class__.__name__,err.shape)

    def train(self, input_x, label_y):
        
        self.label_y = label_y
        self.logits = self.inference(input_x)
        self.update_weigth_bias(self.logits, self.label_y)
        
    def get_loss(self):

        if self.cnn[-1].__class__.__name__ == 'Sigmoidlayer':
            loss = np.sum((self.logits-self.label_y)**2)/self.label_y.shape[0]
        elif self.cnn[-1].__class__.__name__ == 'Softmaxlayer':
            loss = -np.sum(self.label_y*np.log(self.logits))/self.label_y.shape[0]
        return loss

    def get_accuracy(self):

        acc = np.mean((np.equal(np.argmax(self.logits,axis=1), np.argmax(self.label_y,axis=1))))
        return acc


#mnist test

def mnist_test():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    net = Net()
    net.add(Convlayer([1,32,5,5], [1,1], 0.001, padding='SAME'))
    net.add(Maxpoollayer([2,2],[2,2],padding='SAME'))
    net.add(Convlayer([32,64,5,5], [1,1], 0.001, padding='SAME'))
    net.add(Maxpoollayer([2,2],[2,2], padding='SAME'))
    net.add(FClayer([7*7*64, 1024], 0.001))
    net.add(Relulayer())
    net.add(FClayer([1024,10],0.001))
    net.add(Softmaxlayer())
    total_batch = mnist.train._num_examples//32
    print(total_batch)
    for i in range(10):
        for j in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(32)
            batch_x = batch_x.reshape([-1, 1, 28 ,28])
            net.train(batch_x,batch_y)
            if j%10 == 0:
                print(net.get_loss(), net.get_accuracy())

#checking grad

def get_loss(cw, cb, fw, fb, inp, out):
    net = Net()

    conv = Convlayer([2,2,2,2], [2,2], padding='SAME')
    conv.weight = cw.copy()
    conv.bias = cb.copy()
    net.add(conv)
    net.add(Maxpoollayer([2,2],[2,2]))
    fc = FClayer([8,2])
    fc.weight = fw.copy()
    fc.bias = fb.copy()
    net.add(fc)
    net.add(Relulayer())
    net.add(Softmaxlayer())
    net.train(inp, out)
    loss = net.get_loss()
    return loss, net.grads

def grad_check():
    input_x = np.random.random([2,2,8,8])
    output_y = np.array([[1,0],
                         [0,1]])
    delta = 1e-7
    cw_init = np.random.randn(2,2,2,2)
    cb_init = np.zeros([2])
    fw_init = np.random.randn(8,2)
    fb_init = np.zeros([1,2])
    zero = np.zeros([2,2,2,2])
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    w_add = zero.copy()
                    w_add[i,j,m,n] += delta
                    cw_test = cw_init + w_add
                    loss, grad = get_loss(cw_init, cb_init, fw_init, fb_init, input_x, output_y)
                    _loss,_ = get_loss(cw_test, cb_init, fw_init, fb_init, input_x, output_y)
                    grad_true = (_loss-loss)/delta
                    grad_pred = grad[-1][0][i][j][m][n]
                    print("weights({} {} {} {}): actural - expected {} - {}".format(i, j,m,n, grad_true, grad_pred))



grad_check()
# mnist_test()
