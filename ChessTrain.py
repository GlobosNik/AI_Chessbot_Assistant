import chess
import chess.pgn
import zstandard as zstd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
from typing import Dict, List, Tuple, Optional
import io
from tqdm import tqdm

class ChessAI:
    def __init__(self):
        self.model = None
        self.piece_to_int = {
            'White_Pawn': 1, 'White_Rook': 2, 'White_Knight': 3, 
            'White_Bishop': 4, 'White_Queen': 5, 'White_King': 6,
            'Black_Pawn': -1, 'Black_Rook': -2, 'Black_Knight': -3, 
            'Black_Bishop': -4, 'Black_Queen': -5, 'Black_King': -6,
            'board': 0  # Empty square
        }
        self.int_to_piece = {v: k for k, v in self.piece_to_int.items()}
    
    # Convert board dictionary to 8x8 numerical array
    def board_to_vector(self, board_dict: Dict[str, str]) -> np.ndarray:
        board_array = np.zeros((8, 8), dtype=np.int8)
        
        for square_name, piece_class in board_dict.items():
            if square_name == 'turn' or len(square_name) != 2:
                continue
                
            # Convert square name to array indices
            file_idx = ord(square_name[0]) - ord('a')  # a-h -> 0-7
            rank_idx = int(square_name[1]) - 1         # 1-8 -> 0-7
            
            if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                piece_value = self.piece_to_int.get(piece_class, 0)
                board_array[rank_idx, file_idx] = piece_value
        
        return board_array.flatten()
    
    # Convert chessboard to 8x8 numerical array
    def chess_board_to_vector(self, board: chess.Board) -> np.ndarray:
        board_array = np.zeros((8, 8), dtype=np.int8)
        
        piece_map = {
            chess.PAWN: 1, chess.ROOK: 2, chess.KNIGHT: 3,
            chess.BISHOP: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                value = piece_map[piece.piece_type]
                if not piece.color:  # Black pieces are taken as negative
                    value = -value
                board_array[rank, file] = value
        
        return board_array.flatten()
    
    # Convert chess move to output index
    def move_to_output(self, move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square
    
    # Convert output index back to chess move
    def output_to_move(self, output_idx: int) -> chess.Move:
        from_square = output_idx // 64
        to_square = output_idx % 64
        return chess.Move(from_square, to_square)
    
    # Load PGN data and convert to training format
    def load_pgn_data(self, pgn_zst_path: str, max_games: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Loading PGN data from {pgn_zst_path}...")
        
        positions = []
        moves = []
        
        # Decompress and read PGN file
        with open(pgn_zst_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                games_processed = 0
                while games_processed < max_games:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        if board.is_legal(move):
                            # Store position before move
                            positions.append(self.chess_board_to_vector(board))
                            moves.append(self.move_to_output(move))
                            board.push(move)
                    
                    games_processed += 1
                    if games_processed % 100 == 0:
                        print(f"Processed {games_processed} games...")
        
        print(f"Loaded {len(positions)} positions from {games_processed} games")
        return np.array(positions), np.array(moves)
    
    # Create a neural network model
    def create_model(self) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(64,)),  # 8x8 board flattened
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(4096, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, pgn_zst_path: str, model_save_path: str, 
              max_games: int = 10000, epochs: int = 10, batch_size: int = 32):
        
        # Load data
        X, y = self.load_pgn_data(pgn_zst_path, max_games)
        
        # Create and train model
        self.model = self.create_model()
        
        print("Training model...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model and metadata
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.model.save(model_save_path)
        
        # Save piece mapping for later use
        metadata_path = model_save_path.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'piece_to_int': self.piece_to_int,
                'int_to_piece': self.int_to_piece
            }, f)
        
        print(f"Model saved to {model_save_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return history
    
    # Load the trained model
    def load_model(self, model_path: str):
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.piece_to_int = metadata['piece_to_int']
                self.int_to_piece = metadata['int_to_piece']
        
        print(f"Model loaded from {model_path}")
    
    # Predict the best moves for a given board position
    def predictBestMove(self, board_dict: Dict[str, str], top_k: int = 5) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert board to input vector
        board_vector = self.board_to_vector(board_dict)
        board_vector = board_vector.reshape(1, -1)
        
        # Get predictions
        predictions = self.model.predict(board_vector, verbose=0)[0]
        
        # Get top k moves
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            try:
                move = self.output_to_move(idx)
                from_square = chess.square_name(move.from_square)
                to_square = chess.square_name(move.to_square)
                move_str = f"{from_square}{to_square}"
                confidence = float(predictions[idx])
                results.append((move_str, confidence))
            except:
                continue
        
        return results

if __name__ == "__main__":
    chess_ai = ChessAI()
    
    # Training parameters
    PGN_FILE_PATH = "/media/nikhil/BE0F-C323/lichess_db_standard_rated_2018-09.pgn.zst"
    MODEL_SAVE_PATH = "/home/nikhil/Desktop/ChessBot/Globchess.h5"
    
    # Train the model
    print("Starting chess AI training...")
    history = chess_ai.train(
        pgn_zst_path=PGN_FILE_PATH,
        model_save_path=MODEL_SAVE_PATH,
        max_games=5000,
        epochs=20,
        batch_size=64
    )
    
    print("Training completed!")
    
    # Test the prediction functionality
    print("\nTesting prediction...")
    
    # Initialize a sample board position
    example_board = {
        'a1': 'White_Rook', 'b1': 'White_Knight', 'c1': 'White_Bishop', 'd1': 'White_Queen',
        'e1': 'White_King', 'f1': 'White_Bishop', 'g1': 'White_Knight', 'h1': 'White_Rook',
        'a2': 'White_Pawn', 'b2': 'White_Pawn', 'c2': 'White_Pawn', 'd2': 'White_Pawn',
        'e2': 'White_Pawn', 'f2': 'White_Pawn', 'g2': 'White_Pawn', 'h2': 'White_Pawn',
        'a7': 'Black_Pawn', 'b7': 'Black_Pawn', 'c7': 'Black_Pawn', 'd7': 'Black_Pawn',
        'e7': 'Black_Pawn', 'f7': 'Black_Pawn', 'g7': 'Black_Pawn', 'h7': 'Black_Pawn',
        'a8': 'Black_Rook', 'b8': 'Black_Knight', 'c8': 'Black_Bishop', 'd8': 'Black_Queen',
        'e8': 'Black_King', 'f8': 'Black_Bishop', 'g8': 'Black_Knight', 'h8': 'Black_Rook'
    }
    
    # Add empty squares
    for file in 'abcdefgh':
        for rank in '3456':
            example_board[f'{file}{rank}'] = 'board'
    
    # Predict best moves
    best_moves = chess_ai.predictBestMove(example_board, top_k=5)
    print("Top 5 predicted moves:")
    for i, (move, confidence) in enumerate(best_moves, 1):
        print(f"{i}. {move}: {confidence:.4f}")