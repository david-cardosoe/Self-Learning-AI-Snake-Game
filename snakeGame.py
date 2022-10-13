from turtle import Screen
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initializing all pygame modules
pygame.init()

font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    

Point = namedtuple('Point', 'x, y')

# Colors
BLUE = (0, 100, 255)
GREY = (98, 98, 98)
YELLOW = (255, 255, 0)
BLACK = (0,0,0)
RED = (200,0,0)
PINK = (255,188,217)

# Each block's size
BLOCK_SIZE = 20

# Speed snake travels at, is faster since computers don't rely on reaction speed like humans
SPEED = 600

class SnakeGameAI:
    """
    __init__(): This is our constructor where we initialize the width and height.
                Along with w/h we use them to start our display for our game and caption it
                This will get our game started when we call the class in our agent class.
    """
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI')
        self.clock = pygame.time.Clock()
        self.reset()

    """
    reset(): Method resets game when snake dies(current game over)
    """
    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.newFood()
        self.frame_iteration = 0
        
    """
    placeFood(): Method will randomly spawn a new peiece a food for the snake to eat at the start.
                 Will then generate a new peice of food everytime one is eaten by the snake.
    """
    def newFood(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.newFood()
        
    def stepTaken(self, action):
        self.frame_iteration += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self.directionToMove(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False

        """
        The self.frameIteration keeps tracks of the # of frames we have.
        I use this to determine if the snake has gone a long time since eating a apple since 
        we end the game if the frame iteration exceeds 100 * (length of snake). If the snake hasn't eaten an
        apple in a certain amount of fram iterations and frame iteration become > then the limit then we end the game since
        our snake may be caught in a loop.
        """
        #Game over
        if self.collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # If snake hits food incerase the score, set reward to 10 (positive reinforcement), and create a new food
        # for the snake to find
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.newFood()
        else:
            self.snake.pop()
        
        #Updates ui and clock
        self.updateUi()
        self.clock.tick(SPEED)
        # These values returns are gonna be used to see how the snake it doing.
        return reward, game_over, self.score
    
    """
    collision(): Will return true if snake hits the border of our game or if it hits itself, otherwise returns False.
    """
    def collision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    """
    updateUi(): responsible for displaying game green background, white grid, snake itself, apple, as well as the 
                current score of the ai on the current game.
    """

    def updateUi(self):

        self.display.fill(GREY)
        self.drawGrid()
        
        for point in self.snake:
            pygame.draw.rect(self.display, PINK, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(point.x+4, point.y+4, 12, 12))
            
        
        pygame.draw.circle(self.display, RED, [self.food.x + 10, self.food.y + 10], 8) #Draws Food
        
        text = font.render("Score: " + str(self.score), True, YELLOW)
        self.display.blit(text, [5, 15])
        pygame.display.flip()

    """
    drawGrid(): Draws the white grid so we can see exactly how the snake is moving and where the apple is located 
                in the grid.
    """
    def drawGrid(self):
        for x in range(0, 640, BLOCK_SIZE):
            for y in range(0, 480, BLOCK_SIZE):
                rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, BLACK, rect, 1)
        
    
    def directionToMove(self, action):
        # [straight, right, left]

        clockWiseDirections = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clockWiseDirections.index(self.direction)

        """
        Legend:
        [1,0,0] == Straight
        [0,1,0] == Right
        [0,0,1] == Left
        """

        if np.array_equal(action, [1, 0, 0]):
            newDirection = clockWiseDirections[index] #nochange
        elif np.array_equal(action, [0, 1, 0]):
            nextIndex = (index + 1) % 4 #To adjust when we go over the index range of clockWiseDirections
            newDirection = clockWiseDirections[nextIndex] #right Turn r -> d -> l -> u
        else: # [0,0,1]
            nextIndex = (index - 1) % 4
            newDirection = clockWiseDirections[nextIndex] #right Turn r -> u -> l -> d

        self.direction = newDirection

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)