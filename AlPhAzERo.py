import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import pandas as pd
import json
import zipfile
import io
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="AlphaZero Checkers Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👑"
)

st.title("🎯 AlphaZero-Inspired Checkers Arena")
st.markdown("""
Two AI agents battle using AlphaZero-inspired techniques: Monte Carlo Tree Search combined with sophisticated position evaluation.

**AlphaZero Architecture Components:**
- 🌳 **MCTS with UCB** - Monte Carlo Tree Search using Upper Confidence Bounds for exploration/exploitation balance
- 🧠 **Deep Position Evaluation** - Advanced heuristics mimicking neural network evaluation
- 🎯 **Policy & Value Heads** - Dual output system for move selection and position assessment
- 🔄 **Self-Play Training** - Agents improve by playing against themselves
- ⚡ **Minimax with Alpha-Beta** - Strategic depth and tactical precision
- 🎲 **Hybrid Decision Making** - Combining MCTS planning with minimax tactics
""", unsafe_allow_html=True)

# ============================================================================
# American Draughts (Checkers) Environment
# ============================================================================

@dataclass
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    captures: List[Tuple[int, int]]
    promotion: bool = False
    
    def __hash__(self):
        return hash((self.start, self.end, tuple(self.captures)))
    
    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end and 
                self.captures == other.captures)

class Checkers:
    """American Draughts with official rules"""
    def __init__(self):
        self.board_size = 8
        self.reset()
    
    def reset(self):
        """Initialize standard checkers starting position"""
        self.board = np.zeros((8, 8), dtype=int)
        
        # Player 1 (Red) - bottom, moves up
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1
        
        # Player 2 (White) - top, moves down
        for row in range(0, 3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 2
        
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.pieces_count = {1: 12, 2: 12}
        return self.get_state()
    
    def get_state(self):
        """Return hashable board state"""
        return tuple(self.board.flatten())
    
    def copy(self):
        """Deep copy of game state"""
        new_game = Checkers()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        new_game.pieces_count = self.pieces_count.copy()
        return new_game
    
    def get_piece_moves(self, row, col):
        """Get all valid moves for a piece, prioritizing captures"""
        piece = self.board[row, col]
        if piece == 0 or abs(piece) != self.current_player:
            return []
        
        is_king = abs(piece) > 2
        captures = self._get_captures(row, col, is_king)
        
        if captures:
            return captures
        
        # Only return simple moves if no captures available
        return self._get_simple_moves(row, col, is_king)
    
    def _get_simple_moves(self, row, col, is_king):
        """Get non-capturing moves"""
        moves = []
        piece = self.board[row, col]
        
        # Direction based on player
        if piece == 1:  # Red moves up
            directions = [(-1, -1), (-1, 1)]
        elif piece == 2:  # White moves down
            directions = [(1, -1), (1, 1)]
        else:  # Kings move both ways
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < 8 and 0 <= new_col < 8 and 
                self.board[new_row, new_col] == 0):
                promotion = (piece == 1 and new_row == 0) or (piece == 2 and new_row == 7)
                moves.append(Move((row, col), (new_row, new_col), [], promotion))
        
        return moves
    
    def _get_captures(self, row, col, is_king, captured=None):
        """Recursively find all capture sequences (mandatory jumps)"""
        if captured is None:
            captured = []
        
        piece = self.board[row, col]
        moves = []
        
        # All diagonal directions
        if piece == 1:
            directions = [(-1, -1), (-1, 1)]
        elif piece == 2:
            directions = [(1, -1), (1, 1)]
        else:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            jump_row, jump_col = row + dr, col + dc
            land_row, land_col = row + 2*dr, col + 2*dc
            
            if not (0 <= land_row < 8 and 0 <= land_col < 8):
                continue
            
            enemy_piece = self.board[jump_row, jump_col]
            
            # Must jump over enemy piece to empty square
            if (enemy_piece != 0 and 
                (enemy_piece % 3) == (3 - self.current_player) and
                self.board[land_row, land_col] == 0 and
                (jump_row, jump_col) not in captured):
                
                # Multi-jump: continue from landing position
                new_captured = captured + [(jump_row, jump_col)]
                
                # Temporarily make the jump
                original_board = self.board.copy()
                self.board[land_row, land_col] = piece
                self.board[row, col] = 0
                for cr, cc in new_captured:
                    self.board[cr, cc] = 0
                
                # Check for additional jumps
                further_jumps = self._get_captures(land_row, land_col, is_king, new_captured)
                
                self.board = original_board  # Restore
                
                if further_jumps:
                    moves.extend(further_jumps)
                else:
                    # This is a complete capture sequence
                    start = self.move_history[-1].end if captured else (row, col)
                    promotion = ((piece == 1 and land_row == 0) or 
                               (piece == 2 and land_row == 7))
                    moves.append(Move(start, (land_row, land_col), new_captured, promotion))
        
        return moves
    
    def get_all_valid_moves(self):
        """Get all legal moves for current player (captures are forced)"""
        all_moves = []
        has_captures = False
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece != 0 and abs(piece) % 3 == self.current_player:
                    moves = self.get_piece_moves(row, col)
                    if moves and moves[0].captures:
                        has_captures = True
                    all_moves.extend(moves)
        
        # If any captures exist, only return capturing moves
        if has_captures:
            all_moves = [m for m in all_moves if m.captures]
        
        return all_moves
    
    def make_move(self, move: Move):
        """Execute a move and return (next_state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        sr, sc = move.start
        er, ec = move.end
        
        piece = self.board[sr, sc]
        
        # Move piece
        self.board[er, ec] = piece
        self.board[sr, sc] = 0
        
        # Handle captures
        points = 0
        for cr, cc in move.captures:
            captured_piece = self.board[cr, cc]
            self.board[cr, cc] = 0
            points += 3 if abs(captured_piece) > 2 else 1  # Kings worth more
            opponent = 3 - self.current_player
            self.pieces_count[opponent] -= 1
        
        # Promotion to King
        if move.promotion or (piece == 1 and er == 0) or (piece == 2 and er == 7):
            self.board[er, ec] = piece + 2  # 1->3 (Red King), 2->4 (White King)
            points += 2
        
        self.move_history.append(move)
        
        # Check win conditions
        reward = points
        opponent = 3 - self.current_player
        
        if self.pieces_count[opponent] == 0:
            self.game_over = True
            self.winner = self.current_player
            reward = 100
        else:
            # Check if opponent has any moves
            self.current_player = opponent
            if not self.get_all_valid_moves():
                self.game_over = True
                self.winner = 3 - opponent
                self.current_player = 3 - opponent
                reward = 100
        
        return self.get_state(), reward, self.game_over
    
    def evaluate_position(self, player):
        """
        AlphaZero-inspired evaluation function
        Mimics a value head neural network output
        """
        if self.winner == player:
            return 10000
        if self.winner == (3 - player):
            return -10000
        
        opponent = 3 - player
        score = 0
        
        # Material count (like NN feature extraction)
        my_pieces = my_kings = 0
        opp_pieces = opp_kings = 0
        
        # Positional factors
        center_control = 0
        advancement = 0
        back_row_integrity = 0
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece == 0:
                    continue
                
                is_mine = (abs(piece) % 3 == player)
                is_king = abs(piece) > 2
                
                # Material
                if is_mine:
                    if is_king:
                        my_kings += 1
                    else:
                        my_pieces += 1
                else:
                    if is_king:
                        opp_kings += 1
                    else:
                        opp_pieces += 1
                
                # Positional evaluation
                if is_mine:
                    # Center control (squares 2-5, 2-5)
                    if 2 <= row <= 5 and 2 <= col <= 5:
                        center_control += 3 if is_king else 2
                    
                    # Advancement (pushing forward)
                    if piece == 1:  # Red advances upward
                        advancement += (7 - row) * 2
                    elif piece == 2:  # White advances downward
                        advancement += row * 2
                    
                    # Back row integrity (defense)
                    if (player == 1 and row == 7) or (player == 2 and row == 0):
                        back_row_integrity += 5
                else:
                    # Enemy center control
                    if 2 <= row <= 5 and 2 <= col <= 5:
                        center_control -= 3 if is_king else 2
        
        # Scoring weights (tuned for Checkers)
        score += (my_pieces - opp_pieces) * 100
        score += (my_kings - opp_kings) * 300
        score += center_control * 10
        score += advancement
        score += back_row_integrity * 5
        
        # Mobility (move options)
        self.current_player = player
        my_moves = len(self.get_all_valid_moves())
        self.current_player = opponent
        opp_moves = len(self.get_all_valid_moves())
        self.current_player = player
        
        score += (my_moves - opp_moves) * 5
        
        return score

# ============================================================================
# MCTS Node (AlphaZero Core Component)
# ============================================================================

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    def value(self):
        """Average value"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        """Upper Confidence Bound with prior (PUCT algorithm)"""
        if self.visit_count == 0:
            q_value = 0
        else:
            q_value = self.value()
        
        # AlphaZero UCB formula
        u_value = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct=1.5):
        """Select child with highest UCB score"""
        return max(self.children.values(), 
                   key=lambda child: child.ucb_score(self.visit_count, c_puct))
    
    def expand(self, game, policy_priors):
        """Expand node with policy priors (like AlphaZero policy head)"""
        valid_moves = game.get_all_valid_moves()
        
        if not valid_moves:
            return
        
        # Normalize priors
        total_prior = sum(policy_priors.values())
        if total_prior == 0:
            total_prior = len(valid_moves)
        
        for move in valid_moves:
            prior = policy_priors.get(move, 1.0) / total_prior
            child_game = game.copy()
            child_game.make_move(move)
            self.children[move] = MCTSNode(child_game, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def backup(self, value):
        """Backpropagate value up the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(-value)  # Negamax: value flips for opponent

# ============================================================================
# AlphaZero-Inspired Agent
# ============================================================================

class AlphaZeroAgent:
    def __init__(self, player_id, lr=0.3, gamma=0.99, epsilon=1.0):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.96
        self.epsilon_min = 0.01
        
        # Q-Learning component
        self.q_table = {}
        
        # MCTS parameters
        self.mcts_simulations = 250
        self.c_puct = 1.4
        self.minimax_depth = 5
        
        # Policy network (simulated via heuristics)
        self.policy_table = defaultdict(lambda: defaultdict(float))
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        # Training data for self-play
        self.game_history = []
    
    def get_policy_priors(self, game):
        """
        Simulate policy head output using learned preferences and heuristics
        """
        state = game.get_state()
        moves = game.get_all_valid_moves()
        priors = {}
        
        for move in moves:
            # Use learned policy if available
            if state in self.policy_table and move in self.policy_table[state]:
                priors[move] = self.policy_table[state][move]
            else:
                # Heuristic prior: captures are good, center is good
                prior = 1.0
                if move.captures:
                    prior += len(move.captures) * 2
                er, ec = move.end
                if 2 <= er <= 5 and 2 <= ec <= 5:
                    prior += 0.5
                if move.promotion:
                    prior += 1.0
                priors[move] = prior
        
        return priors
    
    def mcts_search(self, game, num_simulations):
        """
        Monte Carlo Tree Search - AlphaZero's planning engine
        """
        root = MCTSNode(game.copy())
        
        for _ in range(num_simulations):
            node = root
            search_game = game.copy()
            search_path = [node]
            
            # Selection: traverse tree using UCB
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_game.make_move(node.move)
                search_path.append(node)
            
            # Expansion: add children if not terminal
            if not search_game.game_over:
                policy_priors = self.get_policy_priors(search_game)
                node.expand(search_game, policy_priors)
            
            # Evaluation: use position evaluation (mimics value head)
            value = self._evaluate_leaf(search_game)
            
            # Backup: propagate value up tree
            node.backup(value)
        
        return root
    
    def _evaluate_leaf(self, game):
        """Evaluate terminal or leaf node (value head simulation)"""
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            elif game.winner == (3 - self.player_id):
                return -1.0
            return 0.0
        
        # Use minimax for deeper evaluation
        
        score = self._minimax(game, self.minimax_depth, -float('inf'), float('inf'), True)
        
        # Normalize to [-1, 1]
        return np.tanh(score / 500)
    
    def _minimax(self, game, depth, alpha, beta, maximizing):
        """Minimax with alpha-beta pruning (tactical component)"""
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)
        
        moves = game.get_all_valid_moves()
        if not moves:
            return game.evaluate_position(self.player_id)
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves[:10]:  # Limit branching
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves[:10]:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def choose_action(self, game, training=True):
        """
        Hybrid decision: MCTS for planning + exploitation
        """
        moves = game.get_all_valid_moves()
        if not moves:
            return None
        
        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(moves)
        
        # Run MCTS
        root = self.mcts_search(game, self.mcts_simulations)
        
        # Select move with highest visit count (AlphaZero method)
        if not root.children:
            return random.choice(moves)
        
        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        # Store visit distribution as policy target
        state = game.get_state()
        total_visits = sum(child.visit_count for child in root.children.values())
        for move, child in root.children.items():
            self.policy_table[state][move] = child.visit_count / total_visits
        
        return best_move
    
    def update_from_game(self, game_data, result):
        """Update agent from completed game (self-play learning)"""
        for state, move, player in game_data:
            if player != self.player_id:
                continue
            
            # Update policy preferences
            if result == self.player_id:
                reward = 1.0
            elif result == 0:
                reward = 0.0
            else:
                reward = -1.0
            
            # Strengthen successful policies
            current_policy = self.policy_table[state][move]
            self.policy_table[state][move] = current_policy + self.lr * (reward - current_policy)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    """Self-play game between two AlphaZero agents"""
    env.reset()
    game_history = []
    agents = {1: agent1, 2: agent2}
    
    move_count = 0
    max_moves = 200
    
    while not env.game_over and move_count < max_moves:
        current_player = env.current_player
        agent = agents[current_player]
        
        state = env.get_state()
        move = agent.choose_action(env, training)
        
        if move is None:
            break
        
        game_history.append((state, move, current_player))
        env.make_move(move)
        move_count += 1
    
    # Update stats and agents
    if env.winner == 1:
        agent1.wins += 1
        agent2.losses += 1
        if training:
            agent1.update_from_game(game_history, 1)
            agent2.update_from_game(game_history, 1)
    elif env.winner == 2:
        agent2.wins += 1
        agent1.losses += 1
        if training:
            agent1.update_from_game(game_history, 2)
            agent2.update_from_game(game_history, 2)
    else:
        agent1.draws += 1
        agent2.draws += 1
        if training:
            agent1.update_from_game(game_history, 0)
            agent2.update_from_game(game_history, 0)
    
    return env.winner

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(board, title="Checkers Board"):
    """Create matplotlib visualization of checkers board"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw checkerboard
    for row in range(8):
        for col in range(8):
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            square = plt.Rectangle((col, 7-row), 1, 1, facecolor=color)
            ax.add_patch(square)
            
            piece = board[row, col]
            if piece != 0:
                # Piece colors
                if abs(piece) % 3 == 1:  # Red
                    piece_color = '#DC143C'
                    edge_color = '#8B0000'
                else:  # White
                    piece_color = '#F5F5F5'
                    edge_color = '#696969'
                
                circle = plt.Circle((col + 0.5, 7-row + 0.5), 0.35, 
                                   color=piece_color, ec=edge_color, linewidth=3)
                ax.add_patch(circle)
                
                # Draw crown for kings
                if abs(piece) > 2:
                    ax.text(col + 0.5, 7-row + 0.5, '♔', 
                           ha='center', va='center', fontsize=24, 
                           color='gold', weight='bold')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Save/Load
# ============================================================================

def create_agents_zip(agent1, agent2, config):
    """Package agents and config into zip"""
    agent1_state = {
        "policy_table": {str(k): {str(m): v for m, v in moves.items()} 
                        for k, moves in agent1.policy_table.items()},
        "epsilon": agent1.epsilon,
        "wins": agent1.wins,
        "losses": agent1.losses,
        "draws": agent1.draws,
        "mcts_sims": agent1.mcts_simulations
    }
    
    agent2_state = {
        "policy_table": {str(k): {str(m): v for m, v in moves.items()} 
                        for k, moves in agent2.policy_table.items()},
        "epsilon": agent2.epsilon,
        "wins": agent2.wins,
        "losses": agent2.losses,
        "draws": agent2.draws,
        "mcts_sims": agent2.mcts_simulations
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(agent1_state))
        zf.writestr("agent2.json", json.dumps(agent2_state))
        zf.writestr("config.json", json.dumps(config))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    """Load agents from uploaded zip file"""
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_state = json.loads(zf.read("agent1.json"))
            agent2_state = json.loads(zf.read("agent2.json"))
            config = json.loads(zf.read("config.json"))
            
            # Reconstruct Agent 1
            agent1 = AlphaZeroAgent(1, 
                                    config.get('lr1', 0.3), 
                                    config.get('gamma1', 0.95))
            agent1.epsilon = agent1_state.get('epsilon', 0.05)
            agent1.wins = agent1_state.get('wins', 0)
            agent1.losses = agent1_state.get('losses', 0)
            agent1.draws = agent1_state.get('draws', 0)
            agent1.mcts_simulations = agent1_state.get('mcts_sims', 150)
            
            # Restore policy table (convert string keys back)
            for state_str, moves_dict in agent1_state.get('policy_table', {}).items():
                state = eval(state_str)
                for move_str, value in moves_dict.items():
                    move = eval(move_str)
                    agent1.policy_table[state][move] = value
            
            # Reconstruct Agent 2
            agent2 = AlphaZeroAgent(2, 
                                    config.get('lr2', 0.3), 
                                    config.get('gamma2', 0.95))
            agent2.epsilon = agent2_state.get('epsilon', 0.05)
            agent2.wins = agent2_state.get('wins', 0)
            agent2.losses = agent2_state.get('losses', 0)
            agent2.draws = agent2_state.get('draws', 0)
            agent2.mcts_simulations = agent2_state.get('mcts_sims', 150)
            
            # Restore policy table
            for state_str, moves_dict in agent2_state.get('policy_table', {}).items():
                state = eval(state_str)
                for move_str, value in moves_dict.items():
                    move = eval(move_str)
                    agent2.policy_table[state][move] = value
            
            return agent1, agent2, config
            
    except Exception as e:
        st.error(f"Failed to load agents: {e}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ AlphaZero Controls")

with st.sidebar.expander("1. Agent 1 (Red) Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.1, 1.0, 0.3, 0.05)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    mcts_sims1 = st.slider("MCTS Simulations₁", 5, 500, 100, 25)
    minimax_depth1 = st.slider("Minimax Depth₁", 1, 10, 1, 1)

with st.sidebar.expander("2. Agent 2 (White) Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.1, 1.0, 0.3, 0.05)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    mcts_sims2 = st.slider("MCTS Simulations₂", 5, 500, 100, 25)
    minimax_depth2 = st.slider("Minimax Depth₂", 1, 10, 1, 1)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 10000, 1000, 100)
    update_freq = st.number_input("Update Every N Games", 10, 500, 50, 10)

with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1:
        config = {
            "lr1": lr1, "gamma1": gamma1, "mcts_sims1": mcts_sims1, "minimax_depth1": minimax_depth1,
            "lr2": lr2, "gamma2": gamma2, "mcts_sims2": mcts_sims2, "minimax_depth2": minimax_depth2,
            "training_history": st.session_state.get('training_history', None)
        }
        
        zip_buffer = create_agents_zip(st.session_state.agent1, st.session_state.agent2, config)
        st.download_button(
            label="💾 Download AlphaZero Agents",
            data=zip_buffer,
            file_name="alphazero_checkers.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Train agents first")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload Saved Agents (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("🔄 Load Agents", use_container_width=True):
            a1, a2, cfg = load_agents_from_zip(uploaded_file)
            if a1 and a2:
                st.session_state.agent1 = a1
                st.session_state.agent2 = a2
                
                # Restore training history if available
                st.session_state.training_history = cfg.get("training_history", None)
                
                st.toast("✅ Agents Loaded Successfully!", icon="🧠")
                st.rerun()
            else:
                st.error("Failed to load agents")

train_button = st.sidebar.button(" Begin Self-Play Training", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Reset Arena", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.rerun()

# Initialize environment and agents
if 'env' not in st.session_state:
    st.session_state.env = Checkers()

if 'agent1' not in st.session_state:
    st.session_state.agent1 = AlphaZeroAgent(1, lr1, gamma1)
    st.session_state.agent1.mcts_simulations = mcts_sims1
    st.session_state.agent1.minimax_depth = minimax_depth1
    
    st.session_state.agent2 = AlphaZeroAgent(2, lr2, gamma2)
    st.session_state.agent2.mcts_simulations = mcts_sims2
    st.session_state.agent2.minimax_depth = minimax_depth2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
env = st.session_state.env

# Update parameters
agent1.mcts_simulations = mcts_sims1
agent1.minimax_depth = minimax_depth1
agent2.mcts_simulations = mcts_sims2
agent2.minimax_depth = minimax_depth2

# Display stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔴 Agent 1 (Red)", 
             f"Policies: {len(agent1.policy_table)}", 
             f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins)
    st.caption(f"MCTS Sims: {agent1.mcts_simulations}")

with col2:
    st.metric("⚪ Agent 2 (White)", 
             f"Policies: {len(agent2.policy_table)}", 
             f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins)
    st.caption(f"MCTS Sims: {agent2.mcts_simulations}")

with col3:
    total = agent1.wins + agent2.wins + agent1.draws
    st.metric("Total Games", total)
    st.metric("Draws", agent1.draws)

st.markdown("---")

# Training
if train_button:
    st.subheader("🎯 AlphaZero Self-Play Training")
    
    status = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [], 'agent2_wins': [], 'draws': [],
        'agent1_epsilon': [], 'agent2_epsilon': [],
        'agent1_policies': [], 'agent2_policies': [],
        'episode': []
    }
    
    for ep in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if ep % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_policies'].append(len(agent1.policy_table))
            history['agent2_policies'].append(len(agent2.policy_table))
            history['episode'].append(ep)
            
            progress = ep / episodes
            progress_bar.progress(progress)
            
            status.markdown(f"""
            | Metric | Agent 1 (Red) | Agent 2 (White) |
            |:-------|:-------------:|:---------------:|
            | **Wins** | {agent1.wins} | {agent2.wins} |
            | **Epsilon** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | **Policies** | {len(agent1.policy_table):,} | {len(agent2.policy_table):,} |
            
            **Game {ep}/{episodes}** ({progress*100:.1f}%) | **Draws:** {agent1.draws}
            """)
    
    progress_bar.progress(1.0)
    st.toast("Training Complete! 🎉", icon="✨")
    st.session_state.training_history = history

# Display training charts
if 'training_history' in st.session_state:
    st.subheader("📊 Training Analytics")
    history = st.session_state.training_history
    df = pd.DataFrame(history)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("#### Win/Draw Distribution")
        chart_data = df[['episode', 'agent1_wins', 'agent2_wins', 'draws']].set_index('episode')
        st.line_chart(chart_data)
    
    with chart_col2:
        st.write("#### Exploration Rate (Epsilon)")
        chart_data = df[['episode', 'agent1_epsilon', 'agent2_epsilon']].set_index('episode')
        st.line_chart(chart_data)
    
    st.write("#### Policy Network Growth")
    chart_data = df[['episode', 'agent1_policies', 'agent2_policies']].set_index('episode')
    st.line_chart(chart_data)

# Final Battle Visualization
if 'agent1' in st.session_state and len(agent1.policy_table) > 100:
    st.subheader("⚔️ Final Championship Match")
    st.info("Watch the trained AlphaZero agents compete in a decisive battle!")
    
    if st.button(" Watch Them Play!", use_container_width=True):
        sim_env = Checkers()
        board_placeholder = st.empty()
        move_text = st.empty()
        
        agents = {1: agent1, 2: agent2}
        move_num = 0
        
        with st.spinner("AlphaZero agents thinking..."):
            while not sim_env.game_over and move_num < 150:
                current_player = sim_env.current_player
                move = agents[current_player].choose_action(sim_env, training=False)
                
                if move is None:
                    break
                
                sim_env.make_move(move)
                move_num += 1
                
                player_name = "Red" if current_player == 1 else "White"
                move_text.caption(f"Move {move_num}: {player_name} plays {move.start} → {move.end}")
                
                fig = visualize_board(sim_env.board, 
                                     f"{player_name}'s Move #{move_num}")
                board_placeholder.pyplot(fig)
                plt.close(fig)
                
                import time
                time.sleep(0.5)
        
        if sim_env.winner == 1:
            st.success("🏆 Agent 1 (Red) Wins!")
        elif sim_env.winner == 2:
            st.error("🏆 Agent 2 (White) Wins!")
        else:
            st.warning("🤝 Draw!")

# ============================================================================
# Human vs AI Arena
# ============================================================================

st.markdown("---")
st.header("🎮 Challenge AlphaZero")

st.markdown("""
<style>
    .game-cell {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70px;
        border-radius: 8px;
        font-size: 36px;
        font-weight: bold;
    }
    .red-piece { color: #DC143C; text-shadow: 0 0 10px rgba(220, 20, 60, 0.5); }
    .white-piece { color: #F5F5F5; text-shadow: 0 0 10px rgba(245, 245, 245, 0.5); }
</style>
""", unsafe_allow_html=True)

if len(agent1.policy_table) > 100:
    col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
    with col_h1:
        opponent = st.selectbox("Your Opponent", ["Agent 1 (Red)", "Agent 2 (White)"])
    with col_h2:
        color_choice = st.selectbox("Your Color", ["Red", "White"])
    with col_h3:
        st.write("")
        if st.button("🎯 Start Game", use_container_width=True, type="primary"):
            st.session_state.human_env = Checkers()
            st.session_state.human_game_active = True
            
            if "Agent 1" in opponent:
                st.session_state.ai_agent = agent1
                st.session_state.ai_player_id = 1 if color_choice == "White" else 2
            else:
                st.session_state.ai_agent = agent2
                st.session_state.ai_player_id = 2 if color_choice == "Red" else 1
            
            st.session_state.human_player_id = 3 - st.session_state.ai_player_id
            st.session_state.selected_piece = None
            st.rerun()
    
    if 'human_env' in st.session_state and st.session_state.human_game_active:
        h_env = st.session_state.human_env
        
        # AI turn
        if h_env.current_player == st.session_state.ai_player_id and not h_env.game_over:
            with st.spinner("🤖 AlphaZero calculating..."):
                import time
                time.sleep(1)
                ai_move = st.session_state.ai_agent.choose_action(h_env, training=False)
                if ai_move:
                    h_env.make_move(ai_move)
                    st.rerun()
        
        # Status
        if h_env.game_over:
            if h_env.winner == st.session_state.human_player_id:
                st.success("🎉 YOU WIN! You defeated AlphaZero!")
            elif h_env.winner == st.session_state.ai_player_id:
                st.error("😮 AlphaZero Wins!")
            else:
                st.warning("🤝 Draw!")
        else:
            turn = "Your Turn" if h_env.current_player == st.session_state.human_player_id else "AI Thinking..."
            st.caption(f"**{turn}**")
        
        # Display board
        fig = visualize_board(h_env.board, "Human vs AlphaZero")
        st.pyplot(fig)
        plt.close(fig)
        
        # Move selection for human
        if (not h_env.game_over and 
            h_env.current_player == st.session_state.human_player_id):
            
            st.write("**Select your piece to move:**")
            valid_moves = h_env.get_all_valid_moves()
            
            # Get unique starting positions
            start_positions = list(set([m.start for m in valid_moves]))
            
            cols = st.columns(min(len(start_positions), 4))
            for idx, pos in enumerate(start_positions):
                if cols[idx % len(cols)].button(f"Piece at {pos}", key=f"select_{pos}"):
                    st.session_state.selected_piece = pos
                    st.rerun()
            
            # Show moves for selected piece
            if 'selected_piece' in st.session_state and st.session_state.selected_piece:
                piece_moves = [m for m in valid_moves if m.start == st.session_state.selected_piece]
                st.write(f"**Moves for piece at {st.session_state.selected_piece}:**")
                
                move_cols = st.columns(min(len(piece_moves), 4))
                for idx, move in enumerate(piece_moves):
                    move_desc = f"→ {move.end}"
                    if move.captures:
                        move_desc += f" (Jump {len(move.captures)})"
                    if move_cols[idx % len(move_cols)].button(move_desc, key=f"move_{idx}"):
                        h_env.make_move(move)
                        st.session_state.selected_piece = None
                        st.rerun()
else:
    st.info("🏋️ Train agents first to unlock Human vs AI mode!")
