import heapq
import datetime
import math

WHITE_FINAL_VAL = 55342202610392170527
BLACK_FINAL_VAL = 112175298107019378335687140669482314051633935764227932350769375443651114565632

"""
" Halma Game state
"""
class halma_game_state:
    def __init__(self, game):
        self.my_pos, self.opp_pos = game.getBoardInfo()
        self.temp_board = 0
        self.board = self.my_pos | self.opp_pos
        self.game = game

    def isTerminal(self):
        """
        " if all pawns in opponent camp
        " return True
        """

        if self.my_pos == WHITE_FINAL_VAL or \
        self.opp_pos == BLACK_FINAL_VAL:
            return True

        return False

    def calculateUtilityValue(self):
        return (self.my_pos | self.opp_pos) - (WHITE_FINAL_VAL | BLACK_FINAL_VAL)

    def calculateHeuristicValue(self):
        return abs(BLACK_FINAL_VAL - self.my_pos)

    def getAllMoves(self, my_turn):
        if my_turn:
            my_pos = self.my_pos
            opp_pos = self.opp_pos
        else:
            my_pos = self.opp_pos
            opp_pos = self.my_pos
        count = 0
        moves = []
        temp1 = my_pos
        while temp1:
            temp2 = temp1
            temp1  = self.game.SetMSBToZero(temp1)
            count += 1
            #print(f'\n\n{count} : {whites:#016b} : {ffs(temp1^temp2)}')
            pos = self.game.ffs(temp1^temp2)
            cur_pos = (pos//16, pos%16)
            print(f'\n\npossible_next_steps for ({15 - cur_pos[0]}, {15 - cur_pos[1]}): ')
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == 15) and (j == 1)):
                            continue
                        else :
                            x = cur_pos[0] + i
                            y = cur_pos[1] + j
                            if self.game.is_pos_valid(x, y):
                                if not ((my_pos | opp_pos) & (1<<(16*x + y))) :
                                    print(f'Single Move to : ({15 - x}, {15 - y})')
                                    moves.append([self.game.calculate_heuristics((x,y)), cur_pos , (x,y)])
                                else:
                                    if self.game.is_pos_valid(x+i, y+j) and not ((my_pos | opp_pos) & (1<<(16*(x+i) + (y+j)))) :
                                        self.temp_board = self.board
                                        moves.extend(self.generate_jump_moves(cur_pos, (cur_pos[0] + 2*i, cur_pos[1] + 2 *j)))
        print(f'Moves: \n {moves}')
        return sorted(moves, key = lambda x : x[0])

    def generate_jump_moves(self, orig_pos, next_jump_pos):
        jump_parent = {}
        jump_parent[orig_pos] = [None, 0]
        jump_parent[next_jump_pos] = [orig_pos, 0]
        moves = []
        open_queue = []
        heapq.heappush(open_queue, (self.game.calculate_heuristics(next_jump_pos), next_jump_pos, orig_pos))
        while open_queue :
            pop_h_val, cur_pos, old_pos = heapq.heappop(open_queue)
            print(f'Jump from : ({15 - old_pos[0]}, {15 - old_pos[1]}) -> ({15 - cur_pos[0]}, {15 - cur_pos[1]})')
            self.set_pawn_position(cur_pos)
            self.unset_pawn_position(old_pos)
            moves.append([pop_h_val, orig_pos, cur_pos])
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == 15) and (j == 1)):
                            continue
                        else :
                            next_pos = (cur_pos[0] + i, cur_pos[1] + j)
                            jump_pos = (next_pos[0] + i, next_pos[1] + j)
                            if self.game.is_pos_valid(next_pos[0], next_pos[1]) and self.game.is_pos_valid(jump_pos[0], jump_pos[1]) and \
                            ((self.temp_board) & (1<<(16*next_pos[0] + next_pos[1]))) and \
                            (not ((self.temp_board) & (1<<(16*jump_pos[0] + jump_pos[1])))) and (jump_pos not in jump_parent):
                                h_val = self.game.calculate_heuristics(jump_pos)
                                #if h_val <= pop_h_val:
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1
        return moves

    def set_pawn_position(self, cur_pos):
        self.temp_board |= (1<<(16*(cur_pos[0]) + cur_pos[1]))

    def unset_pawn_position(self, pos):
        self.temp_board &= ~(1<<(16*(pos[0]) + pos[1]))

    def applyMove(self, move):
        print(f'Apply move : {move}')
        self.board |= (1<<(16*move[1][0] + move[1][1]))
        self.board &= ~(1<<(16*move[0][0] + move[0][1]))

    def unsetMove(self, move):
        self.board |= (1<<(16*move[0][0] + move[0][1]))
        self.board &= ~(1<<(16*move[1][0] + move[1][1]))

"""
" Halma AI agent
"""
class halma_ai_agent:
    def __init__(self, game):
        self.game = game

    def get_nextMove(self):
        state = halma_game_state(self.game)
        nextMove = self.alphaBetaSearch(state, 5)


    def alphaBetaSearch(self, state, depthLimit):
        self.currentDepth = 0
        self.maxDepth = 0
        self.numNodes = 0
        self.maxPruning = 0
        self.minPruning = 0

        self.bestMove = []
        self.depthLimit = depthLimit

        starttime = datetime.datetime.now()
        v = self.maxValue(state, -math.inf, math.inf, self.depthLimit)


        print("Time = " + str(datetime.datetime.now() - starttime))
        print("selected value " + str(v))
        print(f"selected move : {self.bestMove}")
        print("(1) max depth of the tree = {0:d}".format(self.maxDepth))
        print("(2) total number of nodes generated = {0:d}".format(self.numNodes))
        print("(3) number of times pruning occurred in the MAX-VALUE() = {0:d}".format(self.maxPruning))
        print("(4) number of times pruning occurred in the MIN-VALUE() = {0:d}".format(self.minPruning))

        return self.bestMove

    def maxValue(self, state, alpha, beta, depthLimit):
        if state.isTerminal():
            return state.calculateUtilityValue()
        if depthLimit == 0:
            return state.calculateHeuristicValue()

        self.currentDepth += 1
        self.maxDepth = max(self.maxDepth, self.currentDepth)
        self.numNodes += 1

        print(f'{self.currentDepth} : Max turn')
        v = -math.inf
        for move in state.getAllMoves(True)[:2]:
            state.applyMove(move[1:])
            r_val = self.minValue(state, alpha, beta, depthLimit - 1)
            if r_val > v:
                v = r_val
                if depthLimit == self.depthLimit:
                    self.bestMove = move[1:]
            state.unsetMove(move[1:])

            if v >= beta:
                self.maxPruning += 1
                self.currentDepth -= 1
                return v
            alpha = max(alpha, v)

        self.currentDepth -= 1

        return v

    def minValue(self, state, alpha, beta, depthLimit):
        if state.isTerminal():
            return state.calculateUtilityValue()
        if depthLimit == 0:
            return state.calculateHeuristicValue()

        self.currentDepth += 1
        self.maxDepth = max(self.maxDepth, self.currentDepth)
        self.numNodes += 1

        print(f'{self.currentDepth} : Min turn')
        v = math.inf
        for move in state.getAllMoves(False)[:2]:
            state.applyMove(move[1:])
            r_val = self.maxValue(state, alpha, beta, depthLimit - 1)
            if r_val < v:
                v = r_val
            state.unsetMove(move[1:])

            if v <= alpha:
                self.minPruning += 1
                self.currentDepth -= 1
                return v
            beta = min(beta, v)

        self.currentDepth -= 1
        return v

"""
" Main Class
" : Initiate gameplay
"""
class halma_game:

    def __init__(self, game_type, player, play_time, board):
        """
        " Initiate game variables
        """
        self.game_type = game_type
        self.play_time = play_time
        self.player = player
        self.my_pos = 0
        self.opp_pos = 0
        self.temp_board = 0

        self.max_agent = halma_ai_agent(self)
        print(f'Game : {game_type} player : {player} playtime : {play_time}\n board :\n{board}')

        self.board_to_bitboard(board)
        print(f'White_pos : {self.my_pos} black_pos : {self.opp_pos}')

        print(f'Board_pos : {bin(self.my_pos | self.opp_pos)}')

        if self.game_type == "SINGLE":
            self.generate_single_move()
        else:
            self.max_agent.get_nextMove()

        print(f'White_pos : {self.my_pos} black_pos : {self.opp_pos}')

    def board_to_bitboard(self, board):
        """
        " Generate Bitboard from game board
        """
        #pass
        if self.player == "WHITE":
            self.my_pos  = int(board.translate(str.maketrans("W.B", "100")), 2)
            self.opp_pos = int(board.translate(str.maketrans("W.B", "001")), 2)
        else:
            self.opp_pos = int(board.translate(str.maketrans("W.B", "100")), 2)
            self.my_pos  = int(board.translate(str.maketrans("W.B", "001")), 2) 
        
    def bitboard_to_board(self, bit_board):
        """
        " Generate game board from bitboard
        """
        pass

    def getBoardInfo(self):
        return self.my_pos, self.opp_pos

    def generate_single_move(self):
        """
        " Generate single move for a given configuration
        """
        temp1 = self.my_pos
        count = 0 
        while temp1:
            temp2 = temp1
            temp1  = self.SetMSBToZero(temp1)
            count += 1
            #print(f'\n\n{count} : {whites:#016b} : {ffs(temp1^temp2)}')
            pos = self.ffs(temp1^temp2)
            cur_pos = (pos//16, pos%16)
            print(f'\n\npossible_next_steps for ({15 - cur_pos[0]}, {15 - cur_pos[1]}): ')
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:  
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == 15) and (j == 1)):
                            continue
                        else :
                            x = cur_pos[0] + i 
                            y = cur_pos[1] + j 
                            if self.is_pos_valid(x, y):
                                if not ((self.my_pos | self.opp_pos) & (1<<(16*x + y))) :
                                    print(f'Single Move to : ({x}, {y})')
                                    #return
                                else:
                                    if self.is_pos_valid(x+i, y+j) and not ((self.my_pos | self.opp_pos) & (1<<(16*(x+i) + (y+j)))) :
                                        self.generate_jump_moves(cur_pos, (cur_pos[0] + 2*i, cur_pos[1] + 2 *j))

    def generate_jump_moves(self, orig_pos, next_jump_pos):
        jump_parent = {}
        jump_parent[orig_pos] = [None, 0]
        jump_parent[next_jump_pos] = [orig_pos, 0]

        self.temp_board = self.my_pos
        open_queue = []
        heapq.heappush(open_queue, (self.calculate_heuristics(next_jump_pos), next_jump_pos, orig_pos))
        while open_queue :
            pop_h_val, cur_pos, old_pos = heapq.heappop(open_queue)
            print(f'Jump from : ({15 - old_pos[0]}, {15 - old_pos[1]}) -> ({15 - cur_pos[0]}, {15 - cur_pos[1]})')
            self.set_pawn_position(cur_pos)
            self.unset_pawn_position(old_pos)
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:  
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == 15) and (j == 1)):
                            continue
                        else :
                            next_pos = (cur_pos[0] + i, cur_pos[1] + j)
                            jump_pos = (next_pos[0] + i, next_pos[1] + j)
                            if self.is_pos_valid(next_pos[0], next_pos[1]) and self.is_pos_valid(jump_pos[0], jump_pos[1]) and \
                            ((self.temp_board | self.opp_pos) & (1<<(16*next_pos[0] + next_pos[1]))) and \
                            (not ((self.temp_board | self.opp_pos) & (1<<(16*jump_pos[0] + jump_pos[1])))) and (jump_pos not in jump_parent):
                                h_val = self.calculate_heuristics(jump_pos)
                                #if h_val <= pop_h_val:
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1

    def set_pawn_position(self, cur_pos):
        self.temp_board |= (1<<(16*(cur_pos[0]) + cur_pos[1]))

    def unset_pawn_position(self, pos):
        self.temp_board &= ~(1<<(16*(pos[0]) + pos[1]))

    def calculate_heuristics(self, cur_pos):
        d_min = min(abs(cur_pos[0] - 0), abs(cur_pos[1] - 0))
        d_max = max(abs(cur_pos[0] - 0), abs(cur_pos[1] - 0))
        return int(14 * d_min + 10 * (d_max - d_min))

 
    def SetMSBToZero(self, num):
        mask = num 

        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        mask |= mask >> 32
        mask |= mask >> 64
        mask |= mask >> 128 

        mask = mask >> 1

        return num & mask

    def ffs(self, x):
        return (x&-x).bit_length()-1

    def is_pos_valid(self, pos_x, pos_y):
        if (0 <= pos_x <= 15) and (0 <= pos_y <= 15) : return True
    
        return False



if __name__ == "__main__":
    in_list = None
    with open("input.txt", "r") as in_file:
        in_list = in_file.readlines()
    halma_obj = halma_game(in_list[0].strip("\n"), in_list[1].strip("\n"), in_list[2].strip("\n"), "".join(in_list[3:]).replace("\n", ""))
