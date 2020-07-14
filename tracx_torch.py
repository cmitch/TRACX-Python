"""
Created on Sat Feb  6 10:38:20 2016

@author: Caspar Addyman
"""


# module containing several version 1 & 2 of TRACX network
# * Implements the Truncated Recursive Atoassociative Chunk eXtractor
# * (TRACX, French, Addyman & Mareschal, Psych Rev, 2011) A neural network
# * that performs sequence segmentation and chunk extraction in artifical
# * grammar learning tasks and statistical learning tasks.
#
# Note: i've never programmed in Python before so this might get ugly.


import numpy as np
import sys, random
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Tracx_Net(nn.Module):

    def __init__(self, tracx):
        super(Tracx_Net, self).__init__()
        self.tracx = tracx
        if tracx.activation_type == "tanh":
            activate = nn.tanh
        else:
            activate = nn.sigmoid

        self.L1 = nn.Linear(2 * tracx.token_width, tracx.token_width)
        self.A1 = activate()
        self.L2 = nn.Linear(tracx.token_width, 2 * tracx.token_width)
        self.A2 = activate()


    def forward(self, joined_input, prior_d):
        hidden = self.A1(self.L1(joined_input))
        output = self.A2(self.L2(hidden))

        d = output - joined_input

        if self.tracx.delta_rule == "rms":
            d = torch.sqrt(torch.mean(torch.square(d)))
        elif self.tracx.delta_rule == "max":
            d = torch.max(d)
        else:
            raise NotImplementedError("Delta rule not implemented")

        temp = F.tanh(self.tracx.coef_tanh * d)
        if self.tracx.version == 1:
            if prior_d < self.tracx.recognition_criterion:
                new_prior = hidden
            else:
                new_prior = joined_input[:joined_input.size[0]//2]
        else:
            new_prior = (1 - temp) * hidden \
            + temp * joined_input[joined_input.size[0] // 2:]

        return output, new_prior, d

class Tracx():

    def __init__(self, vocabulary, data):

        self.vocabulary = vocabulary
        self.vocab_to_idx = {}
        self.data = data

        self.version = 2

        #default parameters
        self.learning_rate = 0.04
        self.momentum = 0.1

        self.coef_tanh = 0.2197
        self.activation_type = "tanh"
        self.token_width = len(vocabulary)

        self.model = None

        self.recognition_criterion = 0.4
        self.reinforcement_probability = 0.25


        self.encoding_type = "local"   # local,binary,user

        self.delta_rule = "rms"

        #Parameters that were added into dtanh
        self.temperature = 1.0
        self.fahlmanOffset = 0.1

        self.randomSeed = ""     #calculated from string value - leave blank for random
        #TODO: Handle random seed
        # internal variables
        self.trackingInterval = 50
        self.trackingFlag = False
        self.trackingSteps = []
        self.trackingResults = {}
        self.testErrorType = "conditional"


    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model has not been created. Call setup() to create")
        return self._model

    def setup(self):
        if self.encoding_type == "local":
            self.token_width = (np.log(len(self.vocabulary)) / np.log(2)) // 1 + 1
        elif self.encoding_type == "binary"
            self.token_width = len(self.vocabulary)
        else:
            raise NotImplementedError("Encoding method not implemented")

        self.create_input_encodings()

        self._model = Tracx_Net(self)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=self.learning_rate, momentum=self.momentum)


    @encodings.setter
    def encodings(self, new_encodings):
        #return the input encoding for this token
        self._encodings = new_encodings
        #get inputwidth from one of the items        
        self.token_width = len(list(new_encodings.values())[0])


    def create_input_encodings(self):

        #generate the input vectors
        self.encodings = {}
        
        #include blank input as  a zero vector
        self.encodings[" "] = np.zeros(self.token_width) 
        if self.encoding_type == "binary":
            #binary encoding - each  letter numbered and
            #represented by corresponding 8bit binary array of -1 and 1.
            for idx in range(1, len(self.vocabulary) + 1):
                #each input encoded as zeros everywhere
                self.encodings[self.vocabulary[idx - 1]] \
                    = self.decimal_to_binary(idx)
        elif  self.encoding_type == "local":
            #local encoding - one column per letter.
            #i-th column +1, all others -1
            bipolarArray = -1. * np.ones(self.token_width)       
            for idx in range(len(self.vocabulary)):
                bicopy = list(bipolarArray)
                bicopy[idx] = 1.
                #each input encoded as zeros everywhere
                #except for i-th dimension
                self.encodings[self.vocabulary[idx]] = list(bicopy)
        else:
            raise NotImplementedError("Encoding method not implemented")


    def decimal_to_binary(self, n):
        binString = ""
        binArray = []
        bipolarArray = []
        if num >= 2**(self.token_width + 1):
            raise ValueError("Input value too large. Expecting value less than %d" % 2**(self.width +1))

        places = np.arange(self.token_width)[::-1]

        return (n & places) / places

    def train(self, training_data, num_epochs):
        prior = torch.zeros(self.token_width)
        prior_d = float('inf')
        for _ in num_epochs:
            for data in training_data:
                token = data
                new_input = self.encodings[token]

                #TODO: Check if new sentence and handle that case

                joined_input = torch.cat([prior, new_input], dim=0)
                
                self._optimizer.zero_grad()
                output, prior, d = self._model(joined_input)

                if self.version == 2 or d > self.recognition_criterion \
                    or np.random.rand() <= self.reinforcement_probability:
                    loss = self._criterion(output, joined_input)
                    loss.backward()
                    self._optimizer.step()

                    output, prior, d = self._model(joined_input)
                    self._optimizer.zero_grad()


    def test(self, test_data, num_epochs):
        for data in test_data
            inputs, labels = data
            outputs = self._model(inputs)
            loss = self._criterion(outputs, labels)   
        
        
      
    def run_full_simulation(self, printProgress = False):
        self.reset()
        print("Random seed used: " + str(self.randomSeed))
        starttime =  time.time()
        if printProgress:
            print("Simulation started: " + time.strftime("%T",time.localtime()))

        self.currentStep = 0
        inputLength = len(self.trainingData) -1
        self.maxSteps = self.sentenceRepetitions * inputLength  
        testResults =self.create_results_object()
        print( 'Subjects: ')
        # loop round with a new network each time
        for  run_no in range(self.numberSubjects):
            if printProgress:
                print(str(1 + run_no) + ",")            
            if self.trackingFlag:
                # NB tracking data will be overwritten for each new participant                
                # initialise stacked array to store tracking data
                self.trackingSteps = []
                self.trackingResults = {}
                for x in self.trackingWords:
                    self.trackingResults[x] = []           
                                
            self.currentStep = 0
            self.initialize_weights()               
            if self.train_network(-1, printProgress):
                # training worked for this subject
                testResults = self.test_categories(testResults)

        if self.trackingFlag:
            testResults["trackingSteps"] = self.trackingSteps
            testResults["trackingOutputs"] = self.trackingResults
                            
        endtime =  time.time()
        elapsedTime = endtime - starttime
        testResults["elapsedTime"] = elapsedTime
        if printProgress:
            print("Finished. Duration: "+ "{:.3f}".format(elapsedTime) + " secs.")

        return testResults
    

