import heapq
import datetime
import math
import json
import os

BLACK_FINAL_VAL = 55342202610392170527
WHITE_FINAL_VAL = 112175298107019378335687140669482314051633935764227932350769375443651114565632

MIDDLE_BOARD    = 1263795709708793273810665949815186122510200626757953117174276030464


lookup_table = [[((15,12), (13,10)), ((14,11),(12,9)), ((15,13), (11,9)), ((15,15), (13,11)), ((14,14), (10,8)), ((13,12), (12,11)), ((15,14), (9,8))], \
                [((0,3), (2,5)), ((1,4),(3,6)), ((0,2), (4,6)), ((0,0), (2,4)), ((1,1), (5,7)), ((2,3), (3,4)), ((0,1), (6,7))]]
BOARD_SIZE = 16

player_val = 1
inc_val = 0

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
        if self.game.my_player == 0:
            if self.my_pos == WHITE_FINAL_VAL or \
            self.opp_pos == BLACK_FINAL_VAL:
                print(f'0 : Terminated')
                return True
        else:
            if self.my_pos == BLACK_FINAL_VAL or \
            self.opp_pos == WHITE_FINAL_VAL:
                print(f'1: Terminated')
                return True

        return False

    def calculateUtilityValue(self):
        return (self.my_pos | self.opp_pos) - (WHITE_FINAL_VAL | BLACK_FINAL_VAL)

    def calculateHeuristicValue(self, prev_val):
        if self.game.my_player == 0:
            heuristic =  100 * (bin(self.my_pos & WHITE_FINAL_VAL)[2:].count('1') - bin(self.opp_pos & BLACK_FINAL_VAL)[2:].count('1'))
            print(f'1 : calculateHeuristicValue {heuristic}')
            heuristic += 70 * (bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1'))
            print(f"2 : calculateHeuristicValue {bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1')}")
            heuristic += 5 * (self.getDisplacementVal() - prev_val)
            print(f'\n3 WHITES: calculateHeuristicValue {prev_val - self.getDisplacementVal()}')
        else:
            heuristic =  100 * (bin(self.my_pos & BLACK_FINAL_VAL)[2:].count('1') - bin(self.opp_pos & WHITE_FINAL_VAL)[2:].count('1'))
            print(f'1 : calculateHeuristicValue {heuristic}')
            heuristic += 70 * (bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1'))
            print(f"2 : calculateHeuristicValue {bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1')}")
            heuristic += 5 * (prev_val - self.getDisplacementVal())
            print(f'\n3 BLACK: calculateHeuristicValue {prev_val - self.getDisplacementVal()}')

        print(f'\n calculateHeuristicValue {heuristic}')
        return heuristic

    def getDisplacementVal(self):

        temp1 = self.my_pos
        pos = 0
        while temp1:
            temp2 = temp1
            temp1 = self.game.SetMSBToZero(temp1)
            pos     += self.game.ffs(temp1^temp2)
        if self.game.my_player == 0:
            return pos
        else:
            return pos

    def getAllMoves(self, my_turn):
        starttime = datetime.datetime.now()
        if my_turn:
            my_pos = self.my_pos
            opp_pos = self.opp_pos
        else:
            my_pos = self.opp_pos
            opp_pos = self.my_pos

        count = 0
        moves = []
        jump_moves = []
        temp1 = my_pos
        pawn_in_base_camp = False

        if self.game.player == 0:
            if temp1 & BLACK_FINAL_VAL:
                pawn_in_base_camp = True
        else:
            if temp1 & WHITE_FINAL_VAL:
                pawn_in_base_camp = True
            
            
        while temp1:
            temp2 = temp1
            temp1 = self.game.SetMSBToZero(temp1)
            count += 1
            #print(f'\n\n{count} : {whites:#016b} : {ffs(temp1^temp2)}')
            pos     = self.game.ffs(temp1^temp2)
            cur_pos = (pos//BOARD_SIZE, pos%BOARD_SIZE)
            if pawn_in_base_camp and self.move_not_in_my_camp(cur_pos):
                continue
            #print(f'\n\npossible_next_steps for ({(BOARD_SIZE - 1)  - cur_pos[0]}, {(BOARD_SIZE - 1) - cur_pos[1]}): ')
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == BOARD_SIZE - 1) and (j == 1)):
                            continue
                        else :
                            x = cur_pos[0] + i
                            y = cur_pos[1] + j
                            if self.game.is_pos_valid(x, y):
                                if not ((my_pos | opp_pos) & (1<<(BOARD_SIZE*x + y))): #and (self.move_not_in_my_camp((x,y))) :
                                    #print(f'Single Move to : ({BOARD_SIZE - x}, {BOARD_SIZE - y})')
                                    h_val = self.calculate_directional_heuristics((x,y), cur_pos, my_turn)
                                    if h_val > 0:
                                        moves.append([h_val, cur_pos , (x,y)])
                                else:
                                    if self.game.is_pos_valid(x+i, y+j) and ((my_pos | opp_pos) & (1<<(BOARD_SIZE*x + y))) and not ((my_pos | opp_pos) & (1<<(BOARD_SIZE*(x+i) + (y+j)))) :
                                        self.temp_board = self.board
                                        jump_moves.extend(self.generate_jump_moves(cur_pos, (cur_pos[0] + 2*i, cur_pos[1] + 2 *j), my_turn))

        moves = sorted(jump_moves + moves, key = lambda x : x[0], reverse = True)
        print(f'Sorted moves : \n {moves[:10]}')
        print(f'\nlength of moves : {len(moves)}')
        #if my_turn:
            #return sorted(moves, key = lambda x : x[0])[:(len(moves)//4 if len(moves) < 40 else 40)]
        print("Time = " + str(datetime.datetime.now() - starttime))
        return moves
        #else :
            #return sorted(moves, key = lambda x : x[0], reverse = True)[:(len(moves)//4 if len(moves) < 40 else 40)]
            #return sorted(moves, key = lambda x : x[0], reverse = True)[:20]

    def generate_jump_moves(self, orig_pos, next_jump_pos, my_turn):
        jump_parent                = {}
        jump_parent[orig_pos]      = [None, 0]
        jump_parent[next_jump_pos] = [orig_pos, 0]
        moves                      = []
        open_queue                 = []
        heapq.heappush(open_queue, (self.calculate_heuristics(next_jump_pos,orig_pos, my_turn), next_jump_pos, orig_pos))
        while open_queue :
            pop_h_val, cur_pos, old_pos = heapq.heappop(open_queue)
            #print(f'Jump from : ({(BOARD_SIZE - 1) - old_pos[0]}, {(BOARD_SIZE - 1) - old_pos[1]}) -> ({(BOARD_SIZE - 1) - cur_pos[0]}, {(BOARD_SIZE - 1) - cur_pos[1]})')
            self.set_pawn_position(cur_pos)
            self.unset_pawn_position(old_pos)
            h_val = self.calculate_directional_heuristics(cur_pos, orig_pos,  my_turn)
            #if self.move_not_in_my_camp(cur_pos) and h_val > 0:
            if h_val > 0:
                moves.append([h_val, orig_pos, cur_pos])
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == (BOARD_SIZE - 1)) and (j == 1)):
                            continue
                        else :
                            next_pos = (cur_pos[0] + i, cur_pos[1] + j)
                            jump_pos = (next_pos[0] + i, next_pos[1] + j)
                            if self.game.is_pos_valid(next_pos[0], next_pos[1]) and self.game.is_pos_valid(jump_pos[0], jump_pos[1]) and \
                            ((self.temp_board) & (1<<((BOARD_SIZE)*next_pos[0] + next_pos[1]))) and \
                            (not ((self.temp_board) & (1<<((BOARD_SIZE)*jump_pos[0] + jump_pos[1])))) and (jump_pos not in jump_parent):
                                h_val = self.calculate_heuristics(jump_pos, orig_pos, my_turn)
                                #if h_val <= pop_h_val:
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1
        #print(f'moves : {moves}')
        return moves

    def set_pawn_position(self, cur_pos):
        self.temp_board |= (1<<((BOARD_SIZE)*(cur_pos[0]) + cur_pos[1]))

    def unset_pawn_position(self, pos):
        self.temp_board &= ~(1<<((BOARD_SIZE)*(pos[0]) + pos[1]))

    def applyMove(self, move, my_turn):
        print(f'Apply move : {move}')
        if my_turn:
            self.my_pos |= (1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
            self.my_pos &= ~(1<<((BOARD_SIZE)*move[0][0] + move[0][1]))
        else:
            self.opp_pos |= (1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
            self.opp_pos &= ~(1<<((BOARD_SIZE)*move[0][0] + move[0][1]))
        self.board = self.my_pos | self.opp_pos
        #self.print_board_state()

    def unsetMove(self, move, my_turn):
        if my_turn:
            self.my_pos |= (1<<((BOARD_SIZE)*move[0][0] + move[0][1]))
            self.my_pos &= ~(1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
        else:
            self.opp_pos |= (1<<((BOARD_SIZE)*move[0][0] + move[0][1]))
            self.opp_pos &= ~(1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
        self.board = self.my_pos | self.opp_pos

    def calculate_heuristics(self, cur_pos, old_pos, my_turn):
        """end_pos = 0

        if my_turn:
            end_pos = 0
        else:
            end_pos = 15

        d_min = min(abs(cur_pos[0] - end_pos), abs(cur_pos[1] - end_pos)) 
        d_max = max(abs(cur_pos[0] - end_pos), abs(cur_pos[1] - end_pos)) 

        result = int(14 * d_min + 10 * (d_max - d_min))"""

        d_min = min(abs(cur_pos[0] - old_pos[0]), abs(cur_pos[1] - old_pos[1]))
        d_max = max(abs(cur_pos[0] - old_pos[0]), abs(cur_pos[1] - old_pos[1]))
        result = int(1.4 * d_min + (d_max - d_min))

        return result

    def calculate_directional_heuristics(self, cur_pos, old_pos, my_turn):
        #change if condition
        if self.game.player == 0:
            key = ','.join([str(15 - old_pos[0]), str(15 - old_pos[1])])
            if key in self.game.json_data['destined_pos'] :
                end_pos = self.game.json_data['destined_pos'][key]
            else:
                end_pos = (15, 15)
        else:
            end_pos = 0

        if self.game.player == 1:
            d_min = min((old_pos[0] - cur_pos[0]), (old_pos[1] - cur_pos[1]))
            d_max = max((old_pos[0] - cur_pos[0]), (old_pos[1] - cur_pos[1]))
            d_min_goal = min((cur_pos[0]), (cur_pos[1]))
            d_max_goal = max((cur_pos[0]), (cur_pos[1]))
        else:
            d_min = min((cur_pos[0] - old_pos[0]), (cur_pos[1] - old_pos[1]))
            d_max = max((cur_pos[0] - old_pos[0]), (cur_pos[1] - old_pos[1]))
            d_min_goal = min(abs(15 - end_pos[0] - cur_pos[0]), abs(15 - end_pos[1] - cur_pos[1]))
            d_max_goal = max(abs(15 - end_pos[0] - cur_pos[0]), abs(15 - end_pos[1] - cur_pos[1]))
        result = 5 * int(1.4 * d_min + (d_max - d_min)) + int(1.4 * d_min_goal + (d_max_goal - d_min_goal))


        if not self.move_not_in_my_camp(old_pos):
            result += 200

        if not self.move_not_in_my_camp(cur_pos):
            result -= 100

        if self.move_in_opp_camp(cur_pos) and not self.move_in_opp_camp(old_pos):
            result += 100
        elif self.move_in_opp_camp(old_pos) and not self.move_in_opp_camp(cur_pos):
            result -= 200
        elif not self.move_in_opp_camp(old_pos) and self.move_in_midboard(cur_pos):
            result += 75

        # Weight for pawns in the back
        result += 2 * ((15 - cur_pos[0]) + (15 - cur_pos[1]))

        return result

    def move_not_in_my_camp(self, move):
        if self.game.player == 0:
            if (BLACK_FINAL_VAL & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
                return False
        else:
            if (WHITE_FINAL_VAL & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
                return False
        return True


    def move_in_opp_camp(self, move):
        if self.game.player == 0:
            if (WHITE_FINAL_VAL & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
                return True
        else:
            if (BLACK_FINAL_VAL & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
                return True
        return False

    def move_in_midboard(self, move):
        
        if (MIDDLE_BOARD & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
            return True
        
        return False

    def print_board_state(self):
        bin_rep = (format(self.my_pos, '#0258b')[2:] .translate(str.maketrans("10", "W.")))
        cur = 0
        print('my board : \n')
        for i in range(BOARD_SIZE):
            print(bin_rep[cur:cur+BOARD_SIZE])
            cur += BOARD_SIZE
        bin_rep = (format(self.opp_pos, '#0258b')[2:].translate(str.maketrans("10", "B.")))
        cur = 0
        print('Opp board : \n')
        for i in range(BOARD_SIZE):
            print(bin_rep[cur:cur+BOARD_SIZE])
            cur += BOARD_SIZE


"""
" Halma AI agent
"""
class halma_ai_agent:
    def __init__(self, game):
        self.game = game

    def get_nextMove(self):
        state = halma_game_state(self.game)

        num_moves = self.game.json_data['num_moves']
        if num_moves < 7 :
            nextMove = lookup_table[self.game.player][num_moves]
        else:
            nextMove = self.alphaBetaSearch(state, 3)

        return nextMove

    def alphaBetaSearch(self, state, depthLimit):
        self.currentDepth = 0
        self.maxDepth = 0
        self.numNodes = 0
        self.maxPruning = 0
        self.minPruning = 0

        self.bestMove = []
        self.depthLimit = depthLimit

        starttime = datetime.datetime.now()
        cur_val = state.getDisplacementVal()
        v = self.maxValue(state, -math.inf, math.inf, self.depthLimit, cur_val, self.game.player)


        print("Time = " + str(datetime.datetime.now() - starttime))
        print("selected value " + str(v))
        print(f"selected move : ({15 - self.bestMove[0][0]}, {15 - self.bestMove[0][1]}) ---> ({15 - self.bestMove[1][0]}, {15 - self.bestMove[1][1]})")
        print("(1) max depth of the tree = {0:d}".format(self.maxDepth))
        print("(2) total number of nodes generated = {0:d}".format(self.numNodes))
        print("(3) number of times pruning occurred in the MAX-VALUE() = {0:d}".format(self.maxPruning))
        print("(4) number of times pruning occurred in the MIN-VALUE() = {0:d}".format(self.minPruning))

        return self.bestMove

    def maxValue(self, state, alpha, beta, depthLimit, cur_val, player):
        self.game.player = player

        if state.isTerminal():
            return state.calculateUtilityValue()
        if depthLimit == 0:
            return state.calculateHeuristicValue(cur_val)

        self.currentDepth += 1
        self.maxDepth = max(self.maxDepth, self.currentDepth)
        self.numNodes += 1

        #cur_val = state.getDisplacementVal()
        print(f'{self.currentDepth} : Max turn')
        #print(f'cur board : {state.print_board_state()}')
        v = -math.inf
        #print(f' Maxvalue :  moves len : {len(state.getAllMoves(True))}')
        for move in state.getAllMoves(True)[:10]:
            state.applyMove(move[1:], True)
            #cur_val = state.getDisplacementVal()
            r_val = self.minValue(state, alpha, beta, depthLimit - 1, cur_val, (player + 1) % 2)
            if r_val > v:
                v = r_val
                if depthLimit == self.depthLimit:
                    self.bestMove = move[1:]
            state.unsetMove(move[1:], True)

            if v >= beta:
                self.maxPruning += 1
                self.currentDepth -= 1
                return v
            alpha = max(alpha, v)

        self.currentDepth -= 1

        return v

    def minValue(self, state, alpha, beta, depthLimit, cur_val, player):
        self.game.player = player
        if state.isTerminal():
            return state.calculateUtilityValue()
        if depthLimit == 0:
            return state.calculateHeuristicValue(cur_val)

        self.currentDepth += 1
        self.maxDepth = max(self.maxDepth, self.currentDepth)
        self.numNodes += 1

        #cur_val = state.getDisplacementVal()
        print(f'{self.currentDepth} : Min turn')
        #print(f'cur board : {state.print_board_state()}')
        v = math.inf
        #print(f' Minvalue :  moves len : {len(state.getAllMoves(False))}')
        for move in state.getAllMoves(False)[:10]:
            state.applyMove(move[1:], False)
            #cur_val = state.getDisplacementVal()
            r_val = self.maxValue(state, alpha, beta, depthLimit - 1, cur_val, (player+1) % 2)
            if r_val < v:
                v = r_val
            state.unsetMove(move[1:], False)

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

    def __init__(self, game_type, player, play_time, board, num_moves):
        """
        " Initiate game variables
        """
        self.game_type = game_type
        self.play_time = play_time
        
        if player == "WHITE":
            self.player = 0
            self.my_player = 0
        else:
            self.player = 1
            self.my_player = 1
        self.my_pos = 0
        self.opp_pos = 0
        self.temp_board = 0
        self.json_data = []
        self.suggested_move = []

        self.max_agent = halma_ai_agent(self)

        self.board_to_bitboard(board)

        if self.game_type == "SINGLE":

            self.suggested_move = self.generate_single_move()

        else:

            with open("playdata.txt", "r") as data_file:
                self.json_data = json.load(data_file)
            if num_moves == 0:
                self.init_playdata()
            self.suggested_move = self.max_agent.get_nextMove()


        self.moves_completed = num_moves

        self.suggested_move = [[15 - self.suggested_move[0][0], 15 - self.suggested_move[0][1]], [15 - self.suggested_move[1][0], 15 - self.suggested_move[1][1]] ]

        print(f'Suggested move : {self.suggested_move}') 
        self.dump_move()

    def init_playdata(self):

        with open("playdata.txt", "w+") as data_file:
            if 'destined_pos' not in self.json_data:
                if self.my_player  == 1:
                    self.json_data['destined_pos'] = {'0,0' :(15,15), '0,1' :(15,15), '0,2' :(15,15), '0,3' :(15,15), '0,4' :(15,15), '1,0' :(15,15), '1,1' :(15,15), '1,2' :(15,15), '1,3' :(15,15), '1,4' :(15,15), '2,0' :(15,15), '2,1' :(15,15), '2,2' :(15,15), '2,3' :(15,15), '3,0' :(15,15), '3,1' :(15,15), '3,2' :(15,15), '4,0' :(15,15), '4,1' :(15,15)}
                else:
                    self.json_data['destined_pos'] = {'15,15': (0,0), '15,14': (0,0), '15,13': (0,0), '15,12': (0,0), '15,11': (0,0), '14,15': (0,0), '14,14': (0,0), '14,13': (0,0), '14,12': (0,0), '14,11': (0,0), '13,15': (0,0), '13,14': (0,0), '13,13': (0,0), '13,12': (0,0), '12,15': (0,0), '12,14': (0,0), '12,13': (0,0), '11,15': (0,0), '11,14': (0,0)}
            json.dump(self.json_data, data_file)


    def board_to_bitboard(self, board):
        """
        " Generate Bitboard from game board
        """
        if self.player == 0:
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
            pos = self.ffs(temp1^temp2)
            cur_pos = (pos//(BOARD_SIZE), pos%(BOARD_SIZE))
            print(f'\n\npossible_next_steps for ({(BOARD_SIZE - 1) - cur_pos[0]}, {(BOARD_SIZE - 1) - cur_pos[1]}): ')
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:  
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == (BOARD_SIZE - 1)) and (j == 1)):
                            continue
                        else :
                            x = cur_pos[0] + i 
                            y = cur_pos[1] + j 
                            if self.is_pos_valid(x, y):
                                if not ((self.my_pos | self.opp_pos) & (1<<((BOARD_SIZE)*x + y))) :
                                    print(f'Single Move to : ({x}, {y})')
                                    #return
                                else:
                                    if self.is_pos_valid(x+i, y+j) and not ((self.my_pos | self.opp_pos) & (1<<((BOARD_SIZE)*(x+i) + (y+j)))) :
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
            print(f'Jump from : ({(BOARD_SIZE - 1) - old_pos[0]}, {(BOARD_SIZE - 1) - old_pos[1]}) -> ({(BOARD_SIZE - 1) - cur_pos[0]}, {(BOARD_SIZE - 1) - cur_pos[1]})')
            self.set_pawn_position(cur_pos)
            self.unset_pawn_position(old_pos)
            for i in (-1,0,1):
                for j in (-1,0,1):
                    if i != 0 or j != 0:  
                        if ((cur_pos[1] == 0) and (j == -1)):
                            continue
                        elif ((cur_pos[1] == (BOARD_SIZE - 1)) and (j == 1)):
                            continue
                        else :
                            next_pos = (cur_pos[0] + i, cur_pos[1] + j)
                            jump_pos = (next_pos[0] + i, next_pos[1] + j)
                            if self.is_pos_valid(next_pos[0], next_pos[1]) and self.is_pos_valid(jump_pos[0], jump_pos[1]) and \
                            ((self.temp_board | self.opp_pos) & (1<<((BOARD_SIZE)*next_pos[0] + next_pos[1]))) and \
                            (not ((self.temp_board | self.opp_pos) & (1<<((BOARD_SIZE)*jump_pos[0] + jump_pos[1])))) and (jump_pos not in jump_parent):
                                h_val = self.calculate_heuristics(jump_pos)
                                #if h_val <= pop_h_val:
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1

    def set_pawn_position(self, cur_pos):
        self.temp_board |= (1<<((BOARD_SIZE)*(cur_pos[0]) + cur_pos[1]))

    def unset_pawn_position(self, pos):
        self.temp_board &= ~(1<<((BOARD_SIZE)*(pos[0]) + pos[1]))

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
        if (0 <= pos_x <= (BOARD_SIZE - 1)) and (0 <= pos_y <= (BOARD_SIZE - 1)) : return True
    
        return False

    def move_in_opp_camp(self, move):
        if self.my_player == 0:
            if (WHITE_FINAL_VAL & (1<<((BOARD_SIZE)*(15 - move[0]) + 15 - move[1]))):
                return True
        else:
            if (BLACK_FINAL_VAL & (1<<((BOARD_SIZE)*(15 - move[0]) + 15 - move[1]))):
                return True
        return False

    def get_lsb(self, x):
        pos = self.ffs(x)
        return (pos//BOARD_SIZE, pos%BOARD_SIZE)

    def get_msb(self, x):
        pos = self.ffs(x ^ self.SetMSBToZero(x))
        return (pos//BOARD_SIZE, pos%BOARD_SIZE)

        
    def assign_destined_pos(self):
        print(f'Assign Destination positions')
        self.json_data['destined_pos'][','.join(map(str, self.suggested_move[0]))] = self.suggested_move[1]
        if self.my_player == 0:
            my_pos = self.my_pos
            final_pos = WHITE_FINAL_VAL
            for i in range(19):
                key = ','.join(map(str, self.get_lsb(my_pos)))
                self.json_data['destined_pos'][key] = self.get_msb(final_pos)
                my_pos = self.SetMSBToZero(my_pos)
                final_pos = final_pos & ~1
                

    def dump_move(self):

        # Dump move to output.txt

        # Dump data to playdata.txt
        self.json_data = []

        with open("playdata.txt", "r") as data_file:
            self.json_data = json.load(data_file)

        with open("playdata.txt", "w+") as data_file:
            self.json_data['num_moves'] = self.moves_completed + 1
            self.json_data['moves'][','.join(map(str, self.suggested_move[0]))] = self.suggested_move[1]
            self.json_data['destined_pos'][','.join(map(str, self.suggested_move[1]))] = self.json_data['destined_pos'][','.join(map(str, self.suggested_move[0]))]
            del self.json_data['destined_pos'][','.join(map(str, self.suggested_move[0]))]
            if self.move_in_opp_camp(self.suggested_move[1]):
                self.assign_destined_pos()
            json.dump(self.json_data, data_file)

        data_file.close()



if __name__ == "__main__":
    in_list = None
    with open("input.txt", "r") as in_file:
        in_list = in_file.readlines()
    num_moves = 0
    with open("playdata.txt", "r") as playdata_file:
        if os.stat("playdata.txt").st_size == 0:
            num_moves = 0
        else:
            json_data = json.load(playdata_file)
            num_moves = json_data['num_moves']
  
    playdata_file.close()
 
    if num_moves == 0:
        with open("playdata.txt", "w+") as pfile:
            json.dump(json.loads('{"num_moves": 0, "moves": {}}'), pfile)
 
    in_file.close()
    halma_obj = halma_game(in_list[0].strip("\n"), in_list[1].strip("\n"), in_list[2].strip("\n"), "".join(in_list[3:]).replace("\n", ""), num_moves)
