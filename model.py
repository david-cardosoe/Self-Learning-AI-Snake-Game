#from http.client import _DataType
#from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
QNet() is the basis for creating our feed forward neural network, initializes the nn.Module 
    to create neural network with an inputSize, hiddenSize, and ouputSize (3 Layers)

    Layers(Pass in state of snake)
        Will send it into our network and return the output
"""
class QNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linearOne = nn.Linear(inputSize, hiddenSize)
        self.linearTwo = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linearOne(x))
        x = self.linearTwo(x)
        return x


"""
QTrainer becasue we use Deep Q Leanring (Reinforcement learning) to train the snake
    we initialize a learning rate, gamma(discount rate), model, an optimizer, and criterion
    since this class will be used to train the neural network to improve it.

    trainStep()
        trainStep will be called from out agent, will pass in things we need to see how to improve the
        nertwork. I used the bellman equation to train the model based on the reward that is passed in.
        The oldQ and newQ are used to train the model's gradient.
"""
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimimize = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #(n,x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predict Q values with the current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r(reward) + y(gamma) * max(next_predicted Q value) -> only do this if not done
        
        #Resets gradient to 0 so we don't train the model on the old state/gradient, since 
        # we already trained it on that gradient.
        self.optimimize.zero_grad()
        #(QNew, Q)

        loss = self.criterion(target, pred)

        # Computes the gradients when passing backwords through nerual network.
        loss.backward()
        
        self.optimimize.step()

