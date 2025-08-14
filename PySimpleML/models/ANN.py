import numpy as np
import json
import pandas as pd
from ..utils import normalizeDF, deNormalizeDF


class Layer:
    def __init__(self, _neurons = np.zeros((0, 1), dtype=np.float32)):
        self.neurons = _neurons
        self.biases = np.random.rand(self.neurons.shape[0], 1) - 0.5 #random initial biases
        self.dl = np.zeros(_neurons.shape)
        self.weights = np.zeros((0, 1)) #placeholder for weights

class NeuralNetwork:
    # Regression Purpose : Linear Activation [0]
    #Classification : Softmax Activation [1]
    #using MSE cost fucntion
    def __init__(self, *neuronCounts, task=0):
        """
    Creates a Neural Network with random weights and biases
    with mentioned number of neurons per layer.

    Parameters:
        *neuronCounts (int): number of neurons per layer
        mode (int): [0, 1] Type of Supervised ML being adopted. 
        

        Regression: (Mode 0) MSE Loss Fn + ReLU and Linear activation.
        Classification: (Mode 1) Cross-Entropy Loss Fn + ReLU and SoftMax Activation.
    """
        self.task = task
        self.neuronCounts = list(neuronCounts)
    
    def _buildNN(self):
        _layers = np.empty((0, ), dtype=Layer)
        for count in self.neuronCounts:
            neurons = np.zeros((count, 1))
            layer = Layer(neurons)
            _layers = np.append(_layers, layer)
        self.layers = _layers

        for ind in range(1, self.layers.shape[0]):
            self.layers[ind].weights = np.random.rand(self.layers[ind].neurons.shape[0], self.layers[ind-1].neurons.shape[0]) - 0.5
            self.layers[ind].biases = np.random.rand(self.layers[ind].neurons.shape[0], 1) - 0.5

    def fwPass(self, inp: np.ndarray) -> np.float32: #inp should be of shape n*1
        """
        Performs a forward pass with the inputs set to the given NumPy Array.

        Parameter (np.ndarray): input array of shape (n, 1) where n is number of inputs.

        Returns:
            Last layer as a numpy.ndarray of shape (m, 1) where m is the number of outputs.
        """
        def ReLU(x: np.ndarray) -> np.ndarray:
            return x*(x>0)
        
        def LeakyReLU(x: np.ndarray, a = 0.01) -> np.ndarray:
            return np.where(x > 0, x, a * x)
    
        def Softmax(A: np.ndarray) -> np.ndarray:
            A_shifted = A - np.max(A)
            expA = np.exp(A_shifted)
            return expA / np.sum(expA)
        
        # def Sigmoid(x: np.float32) ->np.float32:
        #     return s
        
        def Indentity(x: np.ndarray) -> np.ndarray:
            return x.copy()
        
        def FinalActiv():
            match self.task:
                case 0:
                    return Indentity
                case 1:
                    return Softmax
        
        self.layers[0].neurons = inp
        for ind in range(1, self.layers.shape[0]-1):
            self.layers[ind].z = (np.matmul(self.layers[ind].weights, self.layers[ind-1].neurons) + self.layers[ind].biases)
            self.layers[ind].neurons = LeakyReLU(self.layers[ind].z)
        FinalActivFunc = FinalActiv()
        self.layers[-1].z = (np.matmul(self.layers[-1].weights, self.layers[-2].neurons) + self.layers[-1].biases)
        self.layers[-1].neurons = FinalActivFunc(self.layers[-1].z)
        
        return self.layers[-1].neurons
            
    def bwProp(self, target: np.ndarray, alpha: np.float32): #gets the dls for all neurons
        def ReLUPrime(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 1, dtype=np.float32) * (x>0)
        
        def LeakyReLUPrime(x: np.ndarray, a= 0.01) -> np.ndarray:
            return np.where(x > 0, 1, a)
        
        
        def Sigmoid(x):
            return (1/(1+np.e**(-x)))
        
        def SigmoidPrime(x):
            return Sigmoid(x)*(1-Sigmoid(x))
        
        def SigmoidInverse(x):
            return np.log(x/(1-x))
        
        def DelCSPrime():
            def delMSE(): #Grad Mean Squared Error
                return (target - self.layers[-1].neurons) * LeakyReLUPrime(self.layers[-1].z) 
            
            def delCE(): #Grad Cross Entropy Error 
                return -(self.layers[-1].neurons - target) 
            match self.task:
                case 0:
                    return delMSE()
                case 1:
                    return delCE()
        
        self.layers[-1].dl = -1 * DelCSPrime() 
        for ind in range(self.layers.shape[0] - 2, 0, -1):
            self.layers[ind].dl = np.matmul(self.layers[ind+1].weights.T, self.layers[ind+1].dl) * LeakyReLUPrime(self.layers[ind].z)

        for ind in range(self.layers.shape[0] - 1, 0, -1):
            self.layers[ind].weights -= alpha*np.matmul(self.layers[ind].dl, self.layers[ind-1].neurons.T)
            self.layers[ind].biases -= alpha*self.layers[ind].dl

    def train(self, X, y, alpha: np.float32, epochs: int):
        """
        Trains the Neural Network using the given data.

        Parameters:
            data (numpy.ndarray) : A NumPy array of shape (m, n) where is the number of training examples and n is no. of outputs + no. of inputs.
            alpha (float) : The learning rate.
            iter (int) : No. of epochs.
         """
        X = pd.get_dummies(X, dtype=np.float16)
        #normalizing the inputs
        X, self.xmeans, self.xstds = normalizeDF(X)

        match self.task:
            case 0:
                y, self.ymeans, self.ystds = normalizeDF(y.copy())
                self.ycolumns = list(y.columns)
                self.xcolumns = list(X.columns)
            case 1:
                self.labels = list(y.iloc[:, -1].unique())
                y = pd.get_dummies(y, dtype=np.float16)
                self.ycolumns = list(y.columns)
                self.xcolumns = list(X.columns)

        self.neuronCounts = [len(X.columns)] + self.neuronCounts  + [len(y.columns)]
        self._buildNN()

        X = X.to_numpy().T
        y = y.to_numpy().T

        for epoch in range(epochs):
            for ind in range(X.shape[1]):
                inp = X[:, [ind]]
                self.layers[-1].neurons = self.fwPass(inp)
                self.bwProp(y[:, [ind]], alpha)

        print('Done Training')


    def save(self, name): #saves current weights and biases of the NN
        d = dict()
        d['xstds'] = self.xstds.to_list()
        d['xmeans'] = self.xmeans.to_list()
        d['neuronCounts'] = self.neuronCounts
        d['xcolumns'] = self.xcolumns
        d['ycolumns'] = self.ycolumns
        match self.task:
            case 0:
                d['ymeans'] = self.ymeans.to_list()
                d['ystds'] = self.ystds.to_list()
            case 1:
                d['labels'] = self.labels
        

        for ind in range(1, self.layers.shape[0]):
            d[f'W{ind}'] = self.layers[ind].weights.tolist()
            d[f'B{ind}'] = self.layers[ind].biases.tolist()
        with open(name, 'w') as file:
            json.dump(d, file)

    # def RSq(self, testData):
    #     inps = testData[:, :self.layers[0].neurons.shape[0]]
    #     inps = (inps - inps.mean(axis=0)) / inps.std(axis=0)

    #     ops =  testData[:, self.layers[0].neurons.shape[0]:]
    #     ops = (ops - ops.mean(axis=0)) / ops.std(axis=0)

    #     for x in range(inps.shape[0]):
    #         inp = inps[x, :].reshape(self.layers[0].neurons.shape[0], 1)
    #         target = ops[x, :].reshape(self.layers[-1].neurons.shape[0], 1)
    #         self.op
            
    
    def load(self, name): #loads json file containing weights and biases
        d = dict()        
        with open(name, 'r') as file:
            d = json.load(file)

        
        self.neuronCounts = d['neuronCounts']
        self.xcolumns = d['xcolumns']
        self.ycolumns = d['ycolumns']
        self.xmeans = pd.Series(d['xmeans'], index=self.xcolumns)
        self.xstds = pd.Series(d['xstds'], index=self.xcolumns)
        self._buildNN()

        match self.task:
            case 0:
                self.ymeans = pd.Series(d['ymeans'], index=self.ycolumns)
                self.ystds = pd.Series(d['ystds'], index=self.ycolumns)
                
            case 1:
                self.labels = d['labels']

        for ind in range(1, self.layers.shape[0]):
            self.layers[ind].weights = np.array(d[f'W{ind}'])
            self.layers[ind].biases = np.array(d[f'B{ind}'])
    
    def predict(self, inp: pd.DataFrame, withConfidences=False) -> np.ndarray:
        """
        Returns the predicted output.
        Parameters:
            inp (numpy.ndarray) : A NumPy array of the shape (n, 1) where n is the number of inputs for a forward pass.

        Returns:
            if mode 0: returns a numpy.ndarray that is the output layer.
            if mode 1: returns a tuple with the predicted neuron index and the probability of it.
        """
        inp = pd.get_dummies(inp, dtype=np.float16)
        # print(inp)
        inp = normalizeDF(inp, self.xmeans, self.xstds)[0]
        inp = inp.to_numpy().T
        # print(inp)
        match self.task:
            case 0:
                preds = []
                for ind in range(inp.shape[1]):
                    # print(inp[:, [ind]])
                    opLayer = self.fwPass(inp[:, [ind]]).reshape(-1).tolist()
                    preds.append(opLayer)
                return deNormalizeDF(pd.DataFrame(preds, columns=self.ycolumns), self.ymeans, self.ystds).rename(columns=dict(zip(self.ycolumns, [f'{x}Pred' for x in self.ycolumns])))
                # return self.neuronCounts
            case 1:
                labels = []
                confidences = []
                for ind in range(inp.shape[1]):
                    opLayer = self.fwPass(inp[:, [ind]])
                    labels.append(self.labels[opLayer.argmax()])
                    confidences.append(opLayer.max())
                if withConfidences :
                    op = pd.DataFrame({'Label': labels, 'Confidence': confidences})
                else:
                    op = pd.DataFrame({'Label': labels})
                
                return op