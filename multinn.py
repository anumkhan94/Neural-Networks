# Khan, Anum Farrukh
#This assignment implements a class which can be used to create multi-layer neural networks using Tensorflow.


# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension=input_dimension
        #self.transfer_function=transfer_function
        #self.num_nodes=num_nodes
        #self.layer_number=layer_number
        self.weights=[]
        self.biases=[]
        self.layer_no = 0
        self.layer = {}

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
         "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.layer[self.layer_no] = [num_nodes, transfer_function]
         
        b = tf.Variable(np.random.randn(1,self.layer[self.layer_no][0]))
        self.biases.append(b)
       # print(self.layer)
        #for i in range(len(self.layer)):
        if self.layer_no==0:
            w = tf.Variable(np.random.randn(self.input_dimension,self.layer[self.layer_no][0]))
            self.weights.append(w)
        else:
            print(self.layer[self.layer_no-1][0])
            w = tf.Variable(np.random.randn(self.layer[self.layer_no-1][0],self.layer[self.layer_no][0]))
            self.weights.append(w)
    
        self.layer_no +=1
        


    #def relu(self, x):
     #   return np.maximum(0,x)
        
    #def sigmoid(self, x):
       # return 1/(1+np.exp(-x))

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
         
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]
         

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights
        

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases
        
    #def relu(self, x):
        #return np.maximum(0,x)
        
    #def sigmoid(self, x):
        #return 1/(1+np.exp(-x))

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat)
        return loss
        

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        for layer_no in range(len(self.layer)):
            
            WX = tf.matmul(X,self.weights[layer_no])
            output = WX + self.biases[layer_no]
            #print("Im OP", type(output))
            if self.layer[layer_no][1].lower()=="linear":
                X=output
            elif self.layer[layer_no][1].lower()=="relu":
                X=tf.nn.relu(output)
            elif self.layer[layer_no][1].lower()=="sigmoid":
                #X=self.sigmoid(output)
                X = tf.math.sigmoid(output)
                
        return X


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        # for i in range (num_epochs):
          #  for j in range (0,X.shape[1],batch_size):
                 
         
         



    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        A = self.predict(X)
        return tf.math.confusion_matrix(A, y)
