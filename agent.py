
import torch
import random
import numpy as np
from collections import deque
from snakeGame import SnakeGameAI, Direction, Point
from model import QNet, QTrainer
from graph import plot


MAXMEMORY = 100_000 # Max num states I will hold in memory to train model

SIZE = 2000 # Num states I will use to train the model after every game

LR = 0.001 # Learning rate

"""
Agent class is where the AI will get inforamtion about the environment, send that information or rather state
    to the model do it can train using a reward system (Reinforcment Leanring).
"""
class Agent:
    
    """
    Initiates a few key variables the class needs and that 
        are used throughout the class
    """
    def __init__(self):
        self.numGames = 0 # Number of games the ai has palyed
        self.ran = 0 # used for initial determination of random moves
        self.gamma = 0.9 # Discount Rate 
        self.memory = deque(maxlen=MAXMEMORY) # makes a deque list to hold our memory for training
        self.model = QNet(11, 256, 3) #Creates feed-forward neural network 11(state)-256(Hidden)-3(Direction)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # Creates the class we we use to train our neural network

    """
    Legend:
    S = Staright, R = Right, L = Left, U = Up, D = Down
    
    Environment Inforamtion to get current state of snake so neural network can 
    determine in which direction it should head to next

    States = [Danger S, Danger R, Danger L, dir L, dir R, dir, U, dir D, Food L, Food R, Food U, Food D]
    States = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    0 = False
    1 = True
    """
    def getCurState(self, game):
        head = game.snake[0] #Retrevies Head of snake

        # Holds snake's environment in all 4 directions
        leftPoint = Point(head.x - 20, head.y)
        rightPoint = Point(head.x + 20, head.y)
        upPoint = Point(head.x, head.y - 20)
        downPoint = Point(head.x, head.y + 20)

        # Will tell me which direction snake is cur going
        dirLeft = game.direction == Direction.LEFT
        dirRight = game.direction == Direction.RIGHT
        dirUp = game.direction == Direction.UP
        dirDown = game.direction == Direction.DOWN

        # Format = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        state = [
            # Will turn 1 in the array if there is a danger straight
            (dirRight and game.collision(rightPoint)) or
            (dirLeft and game.collision(leftPoint)) or
            (dirUp and game.collision(upPoint)) or
            (dirDown and game.collision(downPoint)),

            # Will turn 1 in the array if there is a danger right
            (dirUp and game.collision(rightPoint)) or
            (dirDown and game.collision(leftPoint)) or
            (dirLeft and game.collision(upPoint)) or
            (dirRight and game.collision(downPoint)),

            # Will turn 1 in the array if there is a danger left
            (dirDown and game.collision(rightPoint)) or
            (dirUp and game.collision(leftPoint)) or
            (dirRight and game.collision(upPoint)) or
            (dirLeft and game.collision(downPoint)),

            # Will be 1 for whatever direction we are cur moving in
            dirLeft,
            dirRight,
            dirUp,
            dirDown,

            # Will be 1 for whichever direction the food is in on the grid
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]
        
        # Return as numpy array and change True/False to 1/0
        return np.array(state, dtype=int)

    """
    getAction()
        Decides whether snake takes a random move or calculated one using out model.
        I have the snake take random moves initially since the model needs time to train 
        before I can use it to make the snake take calculated moves.
    """
    def getAction(self, state):
    
        self.ran = 60 - self.numGames #Decides when we start training the snake with actual data
        refinedMove = [0,0,0] # Holds final move we take
        
        # generates random move 
        if random.randint(0, 200) < self.ran:
            move = random.randint(0, 2)
            refinedMove[move] = 1
        else: #This move is based on my model
            curState = torch.tensor(state, dtype=torch.float)
    
            #Executes forward function
            prediction = self.model(curState)
            move = torch.argmax(prediction).item() #Get's largets value of netwroks output and use .item() to make it a 1
            refinedMove[move] = 1

        return refinedMove


    # Function will store play given into memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #append as a tuple

    """
    longMemoryTrainer()
        I used this to pass in the tuples of information that we keep in memory to our trainer function, 
        the long memory is only ever trained once the game is over however, not mid game.
    """
    def longMemoryTrainer(self):
        # Grabs 1000 random elements from deque to train model, if there is more than 1000 elements
        if len(self.memory) > SIZE:
            smallSample = random.sample(self.memory, SIZE) # returns a list of tuples
        else:
            smallSample = self.memory # Use entire list if we don't have a > 1000 elements

        states, actions, rewards, nextStates, dones = zip(*smallSample) # Zip makes a combined list for each variable with a list of tuples
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)
        
    """
    shortMemoryTrainer()
        This uses the information gathered at each move and trains the model based on this information
    """
    # Sends our play given to the trainer function in model.py
    def shortMemoryTrainer(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)


"""
trainAI()
    This function initiates the agent and game, starts the entire program. 
    I use the functions I created in this file to run the AI and train the model
    the AI uses to make decisions.
    It will additionally plot the data of how AI is performing and print
    to the console the current game it is on, score of game played, and high score.
"""
def trainAI():
    #Below is the inforamtion we want to keep so we can plot/visualize it
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    #Initialzed the agent and game
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # Get's old state
        oldState = agent.getCurState(game)

        # Calculates next move using the getAction method depedning on the cur state
        refinedMove = agent.getAction(oldState)

        # Snake takes move we determined and gets new state
        reward, done, score = game.stepTaken(refinedMove)
        newState = agent.getCurState(game)

        # Trains short memory of snake.
        # Short memeory meaning training snake baed on prevoius and cur state/move it takes
        agent.shortMemoryTrainer(oldState, refinedMove, reward, newState, done)

        # We will hold this information used to train the short memory in memory to later
        # train the long memory
        agent.remember(oldState, refinedMove, reward, newState, done)

        if done:
            #train the long memory
            # Once we get a game over(done=True), we train long memory below and reset game for snake to play again
            game.reset()
            agent.numGames += 1
            agent.longMemoryTrainer()

            if score > record:
                record = score

            # Prints game # we are on, score we got for that game played, and current high score
            # Will print to terminal
            print('Game:', agent.numGames, 'Score:', score, 'Record:', record)

            # Displays graph with all inforamtion about snake and it's performance
            plotScores.append(score)
            totalScore += score
            mean_score = totalScore / agent.numGames
            plotMeanScores.append(mean_score)
            plot(plotScores, plotMeanScores)

if __name__ == '__main__':
    trainAI()