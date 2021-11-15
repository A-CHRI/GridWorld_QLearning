# Grid World game

# Import libraries used for this program
 
import pygame
import numpy as np

#%%

class GridWorld():    
    # Rendering?
    rendering = False
    
    # Images
    filenames = ['person.png', 'key.png', 'door.png', 'death.png']
    images = [pygame.image.load(file) for file in filenames]

    # Colors
    goodColor = (30, 192, 30)
    badColor = (192, 30, 30)
    pathColor = (225, 220, 225)
    wallColor = (157, 143, 130)
    
    def __init__(self, state=None):
        pygame.init()
        self.reward = 0
        if state is None:
            self.x, self.y, self.has_key, self.board, self.score = self.new_game()            
        else:
            x, y, has_key, board, score = state
            self.x, self.y, self.has_key, self.board, self.score = x, y, has_key, board.copy(), score
    
    def get_state(self):
        return (self.x, self.y, self.has_key)
            
    def step(self, action):
        # Move character
        if not self.game_over(self.x, self.y, self.has_key, self.board):
            self.x, self.y, self.has_key, self.board, self.score, self.reward = self.move(self.x, self.y, self.has_key, self.board, self.score, action)
        
        # return observation, reward, done
        done = self.game_over(self.x, self.y, self.has_key, self.board)        
        return ((self.x, self.y, self.has_key), self.reward, done)
        
    def render(self):
        if not self.rendering:
            self.init_render()
                 
        # Clear the screen
        self.screen.fill((187,173,160))
        
        border = 3
        pygame.draw.rect(self.screen, (187,173,160), pygame.Rect(100,0,600,600))
        for i in range(10):
            for j in range(10):
                val = self.board[i,j]
                col = self.wallColor if val & 8 else self.pathColor
                pygame.draw.rect(self.screen, col, pygame.Rect(100+60*i+border,60*j+border,60-2*border,60-2*border))
                if val>0:
                    x = 105 + 60*i
                    y = 5 + 60*j
                    if val & 4:
                        self.screen.blit(self.images[2], (x, y))
                    if val & 2:
                        self.screen.blit(self.images[1], (x, y))
                    if val & 1:
                        if self.game_over(self.x, self.y, self.has_key, self.board) and not self.won(self.x, self.y, self.has_key, self.board):
                            self.screen.blit(self.images[3], (x, y))
                        else:
                            self.screen.blit(self.images[0], (x, y))
        text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        self.screen.blit(text, (790-text.get_width(), 10))
        
        # Draw game over or you won       
        if self.game_over(self.x, self.y, self.has_key, self.board):
            if self.won(self.x, self.y, self.has_key, self.board):
                msg = 'Congratulations!'
                col = self.goodColor
            else:
                msg = 'Game over!'
                col = self.badColor
            text = self.bigfont.render(msg, True, col)
            textpos = text.get_rect(centerx=self.background.get_width()/2)
            textpos.top = 300
            self.screen.blit(text, textpos)

        # Display
        pygame.display.flip()

    def reset(self):
        self.x, self.y, self.has_key, self.board, self.score = self.new_game()

    def close(self):
        pygame.quit()
                 
    def init_render(self):
        self.screen = pygame.display.set_mode([800, 600])
        pygame.display.set_caption('Grid World')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)
           
    def game_over(self, x, y, has_key, board):
        # Are we on a death square?
        if board[x,y] & 8:
            return True
        
        # Are we on the door with the key?
        if board[x,y] & 4 and not np.any(board & 2):            
            return True
        
        return False
    
    def won(self, x, y, has_key, board):
        # Are we on the door with the key?
        if board[x,y] & 4 and not np.any(board & 2):            
            return True
        
        return False
        
        
    def move(self, x, y, has_key, board, score, direction='left'):
        newx, newy = x, y
        if direction=='left':
            if x>0:
                newx = x-1
        elif direction=='right':
            if x<9:
                newx = x+1
        elif direction=='up':
            if y>0:
                newy = y-1                
        elif direction=='down':
            if y<9:
                newy = y+1
        
        reward = -1
        
        # Update position
        board[x,y] -= 1
        board[newx, newy] += 1
        self.x, self.y = newx, newy
        
        # Take key
        if board[newx, newy] & 2:
            board[newx, newy] -= 2
            reward = 50
            has_key = True
        
        # On door with key?
        if board[newx, newy] & 4 and not np.any(board & 2):
            reward = 100
        
        # On death?
        if board[newx, newy] & 8:
            reward = -100

        score += reward                        
        return (newx, newy, has_key, board, score, reward)
       
    def new_game(self):
        board = np.loadtxt('board.txt', dtype=int).T
        if board.shape != (10,10) or np.sum(board==2) != 1 or np.sum(board==4) != 1:
            raise Exception('board.txt corrupt')

        start_x, start_y = np.where(board == 0)
        i = np.random.randint(len(start_x))
        x, y = start_x[i], start_y[i]
        board[x, y] = 1
        
        score = 0
        has_key = False
        return (x, y, has_key, board, score)


