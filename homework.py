import heapq
import datetime
import math
import json
import os

"""
" WHITE Camp Initial Value
" BLACK Camp Goal Value
"""
BLACK_FINAL_VAL = 55342202610392170527

"""
" BLACK Camp Initial Value
" White Camp Goal Value
"""
WHITE_FINAL_VAL = 112175298107019378335687140669482314051633935764227932350769375443651114565632

"""
" Middle of the board value
"""
MIDDLE_BOARD    = 1263795709708793273810665949815186122510200626757953117174276030464


"""
" Initial moves lookup table
"""
lookup_table = [[((0,3), (2,5)), ((1,4),(3,6)), ((0,2), (4,6)), ((0,0), (2,4)), ((1,1), (5,7)), ((2,3), (3,4)), ((0,1), (6,7))], \
                [((15,12), (13,10)), ((14,11),(12,9)), ((15,13), (11,9)), ((15,15), (13,11)), ((14,14), (10,8)), ((13,12), (12,11)), ((15,14), (9,8))]]

"""
" Size of the board
"""
BOARD_SIZE = 16

"""
" Destination Lookup
"""

destination_lookup = [{'0,0' :(15,15), '0,1' :(15,15), '0,2' :(15,15), '0,3' :(15,15), '0,4' :(15,15), '1,0' :(15,15), '1,1' :(15,15), '1,2' :(15,15), '1,3' :(15,15), '1,4' :(15,15), '2,0' :(15,15), '2,1' :(15,15), '2,2' :(15,15), '2,3' :(15,15), '3,0' :(15,15), '3,1' :(15,15), '3,2' :(15,15), '4,0' :(15,15), '4,1' :(15,15)}, {'15,15': (0,0), '15,14': (0,0), '15,13': (0,0), '15,12': (0,0), '15,11': (0,0), '14,15': (0,0), '14,14': (0,0), '14,13': (0,0), '14,12': (0,0), '14,11': (0,0), '13,15': (0,0), '13,14': (0,0), '13,13': (0,0), '13,12': (0,0), '12,15': (0,0), '12,14': (0,0), '12,13': (0,0), '11,15': (0,0), '11,14': (0,0)}]


"""
" Heuristics weights
" [No of nodes, final_weight, depth, displacement, middle_board, goal, goal_penalty]
"""

game_heuristics = [[40, 150, 3, 5, 70, 100, 50, 4], \
                    [40, 150, 3, 1, 20, 100, 25, 4],
                    [15, 200, 5, 1, 10, 150, 5, 6]]
                    #[40, 200, 3, 1, 10, 150, 5, 6]]
                    #old[40, 200, 3, 1, 10, 120, 5, 6]]

game_stage = 0
"""
" Halma Game state
"""
class halma_game_state:
    def __init__(self, game):
        self.my_pos, self.opp_pos = game.getBoardInfo()
        self.temp_board = 0
        self.board = self.my_pos | self.opp_pos
        self.move_val = 0
        self.game = game

    def isTerminal(self):
        """
        " if all pawns in opponent camp
        " return True
        """
        if self.game.my_player == 0:
            if self.my_pos == WHITE_FINAL_VAL or \
            self.opp_pos == BLACK_FINAL_VAL:
                return True
        else:
            if self.my_pos == BLACK_FINAL_VAL or \
            self.opp_pos == WHITE_FINAL_VAL:
                return True

        return False

    def calculateUtilityValue(self):
        if self.game.my_player == 0:
            if self.my_pos == WHITE_FINAL_VAL:
                return 100000
            elif self.opp_pos == BLACK_FINAL_VAL:
                return -100000
        else:
            if self.my_pos == BLACK_FINAL_VAL:
                return 100000
            elif self.opp_pos == WHITE_FINAL_VAL:
                return -100000

        return 0

    def calculateHeuristicValue(self, prev_val):
        if self.game.my_player == 0:
            heuristic =  game_heuristics[game_stage][5] * (bin(self.my_pos & WHITE_FINAL_VAL)[2:].count('1')) - game_heuristics[game_stage][6] *(bin(self.opp_pos & BLACK_FINAL_VAL)[2:].count('1'))
            #print(f'1 : calculateHeuristicValue {heuristic}')
            heuristic += game_heuristics[game_stage][4] * (bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1'))
            #print(f"2 : calculateHeuristicValue {bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1')}")
            heuristic += game_heuristics[game_stage][3] * (self.getDisplacementVal() - prev_val)
            #print(f'\n3 WHITES: calculateHeuristicValue {prev_val - self.getDisplacementVal()}')
            heuristic += self.move_val

            # Add displacement heuristics for opponent??
        else:
            heuristic =  game_heuristics[game_stage][5] * (bin(self.my_pos & BLACK_FINAL_VAL)[2:].count('1')) 
            heuristic -=  game_heuristics[game_stage][6] * (bin(self.opp_pos & WHITE_FINAL_VAL)[2:].count('1'))
            #print(f'1 : calculateHeuristicValue {heuristic}')
            heuristic += game_heuristics[game_stage][4] * (bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1'))
            #print(f"2 : calculateHeuristicValue {bin(self.my_pos & MIDDLE_BOARD)[2:].count('1') + bin(self.opp_pos & MIDDLE_BOARD)[2:].count('1')}")
            heuristic += game_heuristics[game_stage][3] * (prev_val - self.getDisplacementVal())
            #print(f'\n3 BLACK: calculateHeuristicValue {prev_val - self.getDisplacementVal()}')
            heuristic += self.move_val

        #print(f'\n calculateHeuristicValue {heuristic}')
        return heuristic

    def getDisplacementVal(self):

        temp1 = self.my_pos
        pos = 0
        while temp1:
            temp2 = temp1
            temp1 = self.game.SetMSBToZero(temp1)
            pos     += self.game.ffs(temp1^temp2)

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
        self.moves = []
        self.initial_moves_campout = []
        self.initial_moves_in_camp = []
        temp1 = my_pos
 
        while temp1:
            temp2 = temp1
            temp1 = self.game.SetMSBToZero(temp1)
            count += 1
            pos     = self.game.ffs(temp1^temp2)
            cur_pos = (pos//BOARD_SIZE, pos%BOARD_SIZE)
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
                                    h_val = self.calculate_directional_heuristics((x,y), cur_pos, my_turn)
                                    if not (self.move_not_in_my_camp(cur_pos)) and (self.move_not_in_my_camp((x,y))):
                                        self.initial_moves_campout.append([h_val, cur_pos , (x,y)])
                                    elif not (self.move_not_in_my_camp(cur_pos)) and not (self.move_not_in_my_camp((x,y))) and self.check_valid_move_in_camp(cur_pos, (x, y)) and h_val > 0:
                                        self.initial_moves_in_camp.append([h_val, cur_pos , (x,y)])
                                    else :
                                        self.moves.append([h_val, cur_pos , (x,y)])
                                else:
                                    if self.game.is_pos_valid(x+i, y+j) and ((my_pos | opp_pos) & (1<<(BOARD_SIZE*x + y))) and not ((my_pos | opp_pos) & (1<<(BOARD_SIZE*(x+i) + (y+j)))) :
                                        self.temp_board = self.board
                                        self.generate_jump_moves(cur_pos, (cur_pos[0] + 2*i, cur_pos[1] + 2 *j), my_turn)

        if self.initial_moves_campout:
            return sorted(self.initial_moves_campout, key = lambda x: x[0], reverse = True)
        elif self.initial_moves_in_camp:
            return sorted(self.initial_moves_in_camp, key = lambda x: x[0], reverse = True)
        else:
            return sorted(self.moves, key = lambda x: x[0], reverse = True)

    def generate_jump_moves(self, orig_pos, next_jump_pos, my_turn):
        jump_parent                = {}
        jump_parent[orig_pos]      = [None, 0]
        jump_parent[next_jump_pos] = [orig_pos, 0]
        moves                      = []
        open_queue                 = []
        heapq.heappush(open_queue, (self.calculate_heuristics(next_jump_pos,orig_pos, my_turn), next_jump_pos, orig_pos))
        while open_queue :
            pop_h_val, cur_pos, old_pos = heapq.heappop(open_queue)
            self.set_pawn_position(cur_pos)
            self.unset_pawn_position(old_pos)
            h_val = self.calculate_directional_heuristics(cur_pos, orig_pos,  my_turn)
            if h_val > 0:
                if not (self.move_not_in_my_camp(orig_pos)) and (self.move_not_in_my_camp(cur_pos)):
                    self.initial_moves_campout.append([h_val, orig_pos , cur_pos])
                elif not (self.move_not_in_my_camp(orig_pos)) and not (self.move_not_in_my_camp(cur_pos)) and self.check_valid_move_in_camp(orig_pos, cur_pos):
                    self.initial_moves_in_camp.append([h_val, orig_pos , cur_pos])
                else :
                    self.moves.append([h_val, orig_pos , cur_pos])
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
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1

    def generate_single_jump_moves(self, orig_pos, next_jump_pos, my_turn):
        jump_parent                = {}
        jump_parent[orig_pos]      = [None, 0]
        jump_parent[next_jump_pos] = [orig_pos, 0]
        open_queue                 = []
        heapq.heappush(open_queue, (self.calculate_heuristics(next_jump_pos,orig_pos, my_turn), next_jump_pos, orig_pos))
        while open_queue :
            pop_h_val, cur_pos, old_pos = heapq.heappop(open_queue)
            self.set_pawn_position(cur_pos)
            self.unset_pawn_position(old_pos)
            h_val = self.calculate_directional_heuristics(cur_pos, orig_pos,  my_turn)
            if h_val > 0:
                if not (self.move_not_in_my_camp(orig_pos)) and (self.move_not_in_my_camp(cur_pos)):
                    self.initial_moves_campout.append([h_val, orig_pos , cur_pos])
                elif not (self.move_not_in_my_camp(orig_pos)) and not (self.move_not_in_my_camp(cur_pos)) and self.check_valid_move_in_camp(orig_pos, cur_pos):
                    self.initial_moves_in_camp.append([h_val, orig_pos , cur_pos])
                else :
                    self.moves.append([h_val, orig_pos , cur_pos])
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
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1

    def get_single_move(self):
        my_pos = self.my_pos
        opp_pos = self.opp_pos

        self.moves = []
        self.initial_moves_campout = []
        self.initial_moves_in_camp = []
        self.jump_moves = []
        temp1 = my_pos
    
        while temp1:
            temp2 = temp1
            temp1 = self.game.SetMSBToZero(temp1)
            pos     = self.game.ffs(temp1^temp2)
            cur_pos = (pos//BOARD_SIZE, pos%BOARD_SIZE)

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
                                    h_val = self.get_single_game_heuristics((x,y), cur_pos)
                                    if not (self.move_not_in_my_camp(cur_pos)) and (self.move_not_in_my_camp((x,y))):
                                        self.initial_moves_campout.append([h_val, cur_pos , (x,y)])
                                    elif not (self.move_not_in_my_camp(cur_pos)) and not (self.move_not_in_my_camp((x,y))) and self.check_valid_move_in_camp(cur_pos, (x, y)) and h_val > 0:
                                        self.initial_moves_in_camp.append([h_val, cur_pos , (x,y)])
                                    else :
                                        self.moves.append([h_val, cur_pos , (x,y)])
                                else:
                                    if self.game.is_pos_valid(x+i, y+j) and ((my_pos | opp_pos) & (1<<(BOARD_SIZE*x + y))) and not ((my_pos | opp_pos) & (1<<(BOARD_SIZE*(x+i) + (y+j)))) :
                                        self.temp_board = self.board
                                        self.generate_single_jump_moves(cur_pos, (cur_pos[0] + 2*i, cur_pos[1] + 2 *j), True)

        if self.initial_moves_campout:
            return sorted(self.initial_moves_campout, key = lambda x: x[0], reverse = True)[0][1:]
        elif self.initial_moves_in_camp:
            return sorted(self.initial_moves_in_camp, key = lambda x: x[0], reverse = True)[0][1:]
        else:
            return sorted(self.moves, key = lambda x: x[0], reverse = True)[0][1:]

    def check_valid_move_in_camp(self, old_pos, next_pos):
        if self.game.my_player == 0:
            if (next_pos[0] > old_pos[0]) and (next_pos[1] > old_pos[1]):
                return True
        else:
            if (next_pos[0] < old_pos[0]) and (next_pos[1] <  old_pos[1]):
                return True
        return False

    def set_pawn_position(self, cur_pos):
        self.temp_board |= (1<<((BOARD_SIZE)*(cur_pos[0]) + cur_pos[1]))

    def unset_pawn_position(self, pos):
        self.temp_board &= ~(1<<((BOARD_SIZE)*(pos[0]) + pos[1]))

    def applyMove(self, move, my_turn):
        if my_turn:
            self.move_val += move[0]
            self.my_pos |= (1<<((BOARD_SIZE)*move[2][0] + move[2][1]))
            self.my_pos &= ~(1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
        else:
            self.opp_pos |= (1<<((BOARD_SIZE)*move[2][0] + move[2][1]))
            self.opp_pos &= ~(1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
        self.board = self.my_pos | self.opp_pos
        #print(f'ApplyMove : {move}')
        #self.print_board_state()

    def unsetMove(self, move, my_turn):
        if my_turn:
            self.move_val -= move[0]
            self.my_pos |= (1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
            self.my_pos &= ~(1<<((BOARD_SIZE)*move[2][0] + move[2][1]))
        else:
            self.opp_pos |= (1<<((BOARD_SIZE)*move[1][0] + move[1][1]))
            self.opp_pos &= ~(1<<((BOARD_SIZE)*move[2][0] + move[2][1]))
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

        if self.game.game_type == "SINGLE" and self.game.player == 0:
            end_pos = (15,15)
        elif self.game.game_type == "SINGLE" and self.game.player == 1:
            end_pos = (0,0)

        if self.game.game_type == "GAME" and self.game.player == 0:
            key = ','.join([str(15 - old_pos[0]), str(15 - old_pos[1])])
            if key in self.game.json_data['destined_pos'] :
                end_pos = self.game.json_data['destined_pos'][key]
            else:
                end_pos = (15, 15)
        elif self.game.game_type == "GAME" and self.game.player == 1:
            key = ','.join([str(15 - old_pos[0]), str(15 - old_pos[1])])
            if key in self.game.json_data['destined_pos'] :
                end_pos = self.game.json_data['destined_pos'][key]
            else:
                end_pos = (0,0)

        if self.game.player == 1:
            d_min = min((old_pos[0] - cur_pos[0]), (old_pos[1] - cur_pos[1]))
            d_max = max((old_pos[0] - cur_pos[0]), (old_pos[1] - cur_pos[1]))
        else:
            d_min = min((cur_pos[0] - old_pos[0]), (cur_pos[1] - old_pos[1]))
            d_max = max((cur_pos[0] - old_pos[0]), (cur_pos[1] - old_pos[1]))
        d_min_goal = min(abs(15 - end_pos[0] - cur_pos[0]), abs(15 - end_pos[1] - cur_pos[1]))
        d_max_goal = max(abs(15 - end_pos[0] - cur_pos[0]), abs(15 - end_pos[1] - cur_pos[1]))
        result = 5 * int(1.4 * d_min + (d_max - d_min)) + 1 * int(1.4 * d_min_goal + (d_max_goal - d_min_goal))


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

        if self.move_in_opp_camp(cur_pos) and self.move_in_opp_camp(old_pos):
            if game_stage == 1 or game_stage == 2:
                result -= 75
            else:
                result -= 25

        if self.move_in_opp_camp(cur_pos) and not (self.my_pos & (1<<((BOARD_SIZE)*(cur_pos[0]) + cur_pos[1]))):
            #print(f'Moving from {old_pos} to unoccupied goal {cur_pos}')
            #Change this
            result += game_heuristics[game_stage][1]
        key = ','.join(map(str, old_pos))
        if ((key in self.game.json_data['moves']) and (self.game.json_data['moves'][key] == list(cur_pos))):
            if game_stage == 1 or game_stage == 2:
                #CHANGED THIS FROM 10 to 15!!!!!
                result -= 15
            else:
                result -= 50
        # Weight for pawns in the back
        if not (self.move_in_opp_camp(old_pos)):
            #changed cur_pos to next_pos
            result += game_heuristics[game_stage][7] * ((15 - old_pos[0]) + (15 - old_pos[1]))

        # ADDED GAME STATE CHECK!!!!!
        if self.game.my_player == 0:
            if (old_pos[0] > 7 and cur_pos[0] > 7) and (cur_pos[1] - old_pos[1] > 0):
                result += 25
        elif self.game.my_player == 1:
            if (old_pos[0] < 7 and cur_pos[0] < 7) and (15 - cur_pos[1] -  (15 - old_pos[1]) > 0): 
                result += 25

        return result

    def get_single_game_heuristics(self, cur_pos, old_pos):
        if self.game.player == 1:
            d_min = min((old_pos[0] - cur_pos[0]), (old_pos[1] - cur_pos[1]))
            d_max = max((old_pos[0] - cur_pos[0]), (old_pos[1] - cur_pos[1]))
            d_min_goal = min((cur_pos[0]), (cur_pos[1]))
            d_max_goal = max((cur_pos[0]), (cur_pos[1]))
        else:
            d_min = min((cur_pos[0] - old_pos[0]), (cur_pos[1] - old_pos[1]))
            d_max = max((cur_pos[0] - old_pos[0]), (cur_pos[1] - old_pos[1]))
            d_min_goal = min(abs(15 - cur_pos[0]), abs(15 - cur_pos[1]))
            d_max_goal = max(abs(15 - cur_pos[0]), abs(15 - cur_pos[1]))
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
        #result += 2 * ((15 - cur_pos[0]) + (15 - cur_pos[1]))

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
        #print('my board : \n')
        for i in range(BOARD_SIZE):
            #print(bin_rep[cur:cur+BOARD_SIZE])
            cur += BOARD_SIZE
        bin_rep = (format(self.opp_pos, '#0258b')[2:].translate(str.maketrans("10", "B.")))
        cur = 0
        #print('Opp board : \n')
        for i in range(BOARD_SIZE):
            #print(bin_rep[cur:cur+BOARD_SIZE])
            cur += BOARD_SIZE


"""
" Halma AI agent
"""
class halma_ai_agent:
    def __init__(self, game):
        self.game = game

    def get_nextMove(self):
        global game_stage
        state = halma_game_state(self.game)

        num_moves = self.game.json_data['num_moves']
        if num_moves < 7 :
            nextMove = lookup_table[self.game.my_player][num_moves]
        else:
            self.set_game_stage()
            nextMove = self.alphaBetaSearch(state, game_heuristics[game_stage][2])

        return nextMove


    def set_game_stage(self):
        global game_stage

        if self.game.my_player == 0:

            if bin(self.game.my_pos & WHITE_FINAL_VAL)[2:].count('1') >= 16:
                game_stage = 2
            elif bin(self.game.my_pos & WHITE_FINAL_VAL)[2:].count('1') > 12:
                game_stage = 1
            else:
                game_stage = 0
        elif self.game.my_player == 1:
            
            if bin(self.game.my_pos & BLACK_FINAL_VAL)[2:].count('1') >= 16:
                game_stage = 2
            if bin(self.game.my_pos & BLACK_FINAL_VAL)[2:].count('1') > 12:
                game_stage = 1
            else:
                game_stage = 0

    def generate_single_move(self):
        state = halma_game_state(self.game)

        return state.get_single_move()

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

        print("\nNiroop's Move")
        print("Time = " + str(datetime.datetime.now() - starttime))
        print("selected value " + str(v))
        print(f"selected move : ({15 - self.bestMove[0][0]}, {15 - self.bestMove[0][1]}) ---> ({15 - self.bestMove[1][0]}, {15 - self.bestMove[1][1]})")
        print("(1) max depth of the tree = {0:d}".format(self.maxDepth))
        print("(2) total number of nodes generated = {0:d}".format(self.numNodes))
        print("(3) number of times pruning occurred in the MAX-VALUE() = {0:d}".format(self.maxPruning))
        print("(4) number of times pruning occurred in the MIN-VALUE() = {0:d}".format(self.minPruning))
        print("\n\n")

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
        #print(f'{self.currentDepth} : Max turn')
        #print(f'cur board : {state.print_board_state()}')
        v = -math.inf
        #print(f' Maxvalue :  moves len : {len(state.getAllMoves(True))}')
        for move in state.getAllMoves(True)[:game_heuristics[game_stage][0]]:
            state.applyMove(move, True)
            #cur_val = state.getDisplacementVal()
            r_val = self.minValue(state, alpha, beta, depthLimit - 1, cur_val, (player + 1) % 2)
            if r_val > v:
                v = r_val
                if depthLimit == self.depthLimit:
                    #print(f'Max selected : {move[1:]}')
                    self.bestMove = move[1:]
            state.unsetMove(move, True)

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
        #print(f'{self.currentDepth} : Min turn')
        #print(f'cur board : {state.print_board_state()}')
        v = math.inf
        #print(f' Minvalue :  moves len : {len(state.getAllMoves(False))}')
        for move in state.getAllMoves(False)[:game_heuristics[game_stage][0]]:
            state.applyMove(move, False)
            #cur_val = state.getDisplacementVal()
            r_val = self.maxValue(state, alpha, beta, depthLimit - 1, cur_val, (player+1) % 2)
            if r_val < v:
                v = r_val
            state.unsetMove(move, False)

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

            self.suggested_move = self.max_agent.generate_single_move()

        else:
            num_moves = 0
            with open("playdata_n.txt", "r") as playdata_file:
                if os.stat("playdata_n.txt").st_size == 0:
                    num_moves = 0
                else:
                    self.json_data = json.load(playdata_file)
                    num_moves = self.json_data['num_moves']

                playdata_file.close()

            if num_moves == 0:
                with open("playdata_n.txt", "w+") as pfile:
                    self.json_data = json.loads('{"num_moves": 0, "moves": {}}')
                    self.init_playdata()
            self.suggested_move = self.max_agent.get_nextMove()

            self.moves_completed = num_moves

        print(f'Suggested Move : {self.suggested_move}')
        self.dump_move()

    def init_playdata(self):
        """
        " Initialize Playdata file
        """
        with open("playdata_n.txt", "w+") as data_file:
            if 'destined_pos' not in self.json_data:
                if self.my_player  == 0:
                    self.json_data['destined_pos'] = destination_lookup[1]
                else:
                    self.json_data['destined_pos'] = destination_lookup[0]
            json.dump(self.json_data, data_file)


    def board_to_bitboard(self, board):
        """
        " Generate Bitboard from game board
        """
        if self.my_player == 0:
            self.my_pos  = int(board.translate(str.maketrans("W.B", "100")), 2)
            self.opp_pos = int(board.translate(str.maketrans("W.B", "001")), 2)
        else:
            self.opp_pos = int(board.translate(str.maketrans("W.B", "100")), 2)
            self.my_pos  = int(board.translate(str.maketrans("W.B", "001")), 2) 
        
    def getBoardInfo(self):
        """
        " Get positions of the both the players
        """
        return self.my_pos, self.opp_pos

    def set_pawn_position(self, cur_pos):
        self.temp_board |= (1<<((BOARD_SIZE)*(cur_pos[0]) + cur_pos[1]))

    def unset_pawn_position(self, pos):
        self.temp_board &= ~(1<<((BOARD_SIZE)*(pos[0]) + pos[1]))

    def calculate_heuristics(self, cur_pos, next_pos):
        d_min = min(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
        d_max = max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
        return int(1.4 * d_min + (d_max - d_min))

 
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
            if (WHITE_FINAL_VAL & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
                return True
        else:
            if (BLACK_FINAL_VAL & (1<<((BOARD_SIZE)*(move[0]) + move[1]))):
                return True
        return False

    def get_lsb(self, x):
        pos = self.ffs(x)
        return (pos//BOARD_SIZE, pos%BOARD_SIZE)

    def get_msb(self, x):
        pos = self.ffs(x ^ self.SetMSBToZero(x))
        return (pos//BOARD_SIZE, pos%BOARD_SIZE)

    def apply_suggested_move(self):
        self.my_pos |= (1<<((BOARD_SIZE)*self.suggested_move[1][0] + self.suggested_move[1][1]))
        self.my_pos &= ~(1<<((BOARD_SIZE)*self.suggested_move[0][0] + self.suggested_move[0][1]))

        self.board = self.my_pos | self.opp_pos

    def assign_destined_pos(self):

        self.apply_suggested_move()
        if self.my_player == 0:
            my_pos = self.my_pos
            final_pos = BLACK_FINAL_VAL
        else:
            final_pos = WHITE_FINAL_VAL
        my_pos = self.my_pos
        for i in range(19):
            pos_msb = self.get_msb(my_pos)
            final_msb = self.get_msb(final_pos)
            key = ','.join(map(str, pos_msb))
            self.json_data['destined_pos'][key] = final_msb
            my_pos = self.SetMSBToZero(my_pos)
            final_pos = self.SetMSBToZero(final_pos)

    def get_path(self, target_pos, visited_dict):
        path_list  = []

        temp_pos  = target_pos
        while temp_pos is not None:
            path_list.insert(0,  str(15 - temp_pos[1]) + "," + str(15 - temp_pos[0]))
            temp_pos = visited_dict[temp_pos[0:2]][0]

        return path_list

    def generate_jump_path(self, orig_pos, goal):
        jump_parent = {}
        jump_parent[orig_pos] = [None, 0]

        self.temp_board = self.my_pos | self.opp_pos
        open_queue = []
        heapq.heappush(open_queue, (0, orig_pos, None))
        while open_queue :
            pop_h_val, cur_pos, old_pos = heapq.heappop(open_queue)
            if cur_pos == goal:
                return self.get_path(goal, jump_parent)
            self.set_pawn_position(cur_pos)
            if  old_pos:    
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
                            ((self.temp_board) & (1<<((BOARD_SIZE)*next_pos[0] + next_pos[1]))) and \
                            (not ((self.temp_board) & (1<<((BOARD_SIZE)*jump_pos[0] + jump_pos[1])))) and (jump_pos not in jump_parent):
                                h_val = self.calculate_heuristics(cur_pos, jump_pos)
                                heapq.heappush(open_queue, (h_val, jump_pos, cur_pos))
                                jump_parent[jump_pos] = [cur_pos, 0]
            jump_parent[cur_pos][1] = 1
        
    def generate_path(self):
        goal = tuple(self.suggested_move[1])
        count = 0
        cur_pos = tuple(self.suggested_move[0])
        o_path = ()
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
                            if not ((self.my_pos | self.opp_pos) & (1<<((BOARD_SIZE)*x + y))) and (x,y) == goal:
                                return (0, self.suggested_move)
                            else:
                                if self.is_pos_valid(x+i, y+j) and ((self.my_pos | self.opp_pos) & (1<<(BOARD_SIZE*x + y))) and not ((self.my_pos | self.opp_pos) & (1<<((BOARD_SIZE)*(x+i) + (y+j)))) :
                                    #return 1, self.generate_jump_path(cur_pos, (cur_pos[0] + 2*i, cur_pos[1] + 2 *j), goal)
                                    o_path = (1, self.generate_jump_path(cur_pos,goal))
                                    if o_path[1] is not None:
                                        return o_path
        return o_path
        

    def dump_move(self):

        # Dump move to output.txt

        move_type, move_path = self.generate_path()

        print(f'Move Suggested : {self.suggested_move}')
        with open("output.txt", "w+") as o_file:
            if move_type == 0:
                move_path = [[15 - move_path[0][0], 15 - move_path[0][1]], [15 - move_path[1][0], 15 - move_path[1][1]] ]
                o_file.write(f"E {','.join(list(map(str, move_path[0][::-1])))} {','.join(list(map(str, move_path[1][::-1])))}")
            else:
                idx = 0
                o_file.write("\n".join(["J " + move_path[idx] + " " + move_path[idx + 1] for idx in range(len(move_path) - 1)]))

        o_file.close()

        # Dump data to playdata.txt
        if self.game_type == "GAME":
            with open("playdata_n.txt", "r") as data_file:
                self.json_data = json.load(data_file)

            self.suggested_move = [[15 - self.suggested_move[0][0], 15 - self.suggested_move[0][1]], [15 - self.suggested_move[1][0], 15 - self.suggested_move[1][1]]]
            with open("playdata_n.txt", "w+") as data_file:
                self.json_data['num_moves'] = self.moves_completed + 1
                self.json_data['moves'][','.join(map(str, self.suggested_move[0]))] = self.suggested_move[1]
                self.json_data['destined_pos'][','.join(map(str, self.suggested_move[1]))] = self.json_data['destined_pos'][','.join(map(str, self.suggested_move[0]))]
                del self.json_data['destined_pos'][','.join(map(str, self.suggested_move[0]))]
                if self.move_in_opp_camp(self.suggested_move[1]):
                    self.assign_destined_pos()
                json.dump(self.json_data, data_file)

            data_file.close()

        print(f'Suggested move : {self.suggested_move}')

if __name__ == "__main__":
    in_list = None
    with open("input.txt", "r") as in_file:
        in_list = in_file.readlines()
    in_file.close()
    halma_obj = halma_game(in_list[0].strip("\n"), in_list[1].strip("\n"), in_list[2].strip("\n"), "".join(in_list[3:]).replace("\n", ""))
