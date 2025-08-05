import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle

import numpy as np
import os
import time
from PIL import Image as PILImage

import boardDetector
from boardDetector import ChessboardDetector

try:
    from ChessTrain import ChessAI
    CHESS_AI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ChessAITrainer: {e}")
    CHESS_AI_AVAILABLE = False

kivy.require('2.0.0')

# Configuration paths
pieceDetector_path = '/home/nikhil/Desktop/ChessBot/chessPieceDetector.pt'
movePredictor_path = '/home/nikhil/Desktop/ChessBot/Globchess.h5'

class TurnSelectionPopup(Popup):
    def __init__(self, callback, **kwargs):
        super(TurnSelectionPopup, self).__init__(**kwargs)
        self.title = 'Whose Turn to Move?'
        self.size_hint = (0.6, 0.5)
        self.callback = callback
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Question label
        question_label = Label(
            text='Select whose turn it is to move:',
            font_size=18,
            font_name='Roboto',
            size_hint_y=0.3
        )
        layout.add_widget(question_label)
        
        # Button layout
        button_layout = BoxLayout(orientation='horizontal', spacing=20, size_hint_y=0.4)
        
        # White's turn button
        white_button = Button(
            text="White's Turn",
            font_size=16,
            font_name='Roboto',
            background_color=(0.9, 0.9, 0.9, 1),
            color=(0, 0, 0, 1)
        )
        white_button.bind(on_release=lambda x: self.select_turn('white'))
        button_layout.add_widget(white_button)
        
        # Black's turn button
        black_button = Button(
            text="Black's Turn",
            font_size=16,
            font_name='Roboto',
            background_color=(0.2, 0.2, 0.2, 1),
            color=(1, 1, 1, 1)
        )
        black_button.bind(on_release=lambda x: self.select_turn('black'))
        button_layout.add_widget(black_button)
        
        layout.add_widget(button_layout)
        
        # Cancel button
        cancel_button = Button(
            text='Cancel',
            font_size=14,
            font_name='Roboto',
            size_hint_y=0.3,
            background_color=(0.5, 0.5, 0.5, 1)
        )
        cancel_button.bind(on_release=self.dismiss)
        layout.add_widget(cancel_button)
        
        self.content = layout
    
    def select_turn(self, turn):
        self.callback(turn)
        self.dismiss()

class BGWidget(BoxLayout):
    def __init__(self, default_image_path='/home/nikhil/Desktop/ChessBot/icon.png', **kwargs):
        super(BGWidget, self).__init__(**kwargs)
        self.image = Image(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.image)

        self.default_image_path = default_image_path
        self.set_source(self.default_image_path)
        self.bind(size=self.update_background, pos=self.update_background)

    def show_black_background(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 0, 0, 1)
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)

    def hide_black_background(self):
        self.canvas.before.clear()

    def update_background(self, *args):
        if hasattr(self, 'bg_rect'):
            self.bg_rect.size = self.size
            self.bg_rect.pos = self.pos

    def set_source(self, source_path):
        if source_path and os.path.exists(source_path):
            self.image.source = source_path
            self.image.reload()
            self.hide_black_background()
        else:
            self.image.source = ''
            self.show_black_background()

    def clear_image(self):
        self.image.source = ''
        self.show_black_background()

    def reload(self):
        self.image.reload()


class Globchess(BoxLayout):
    def __init__(self, **kwargs):
        super(Globchess, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text='GLOBCHESS: THE CHESS AI ASSISTANT', size_hint_y=0.1, font_size=24, font_name='Roboto', color=(1, 1, 1, 1)))

        self.board_detector = ChessboardDetector(pieceDetector_path)

        if CHESS_AI_AVAILABLE:
            self.chess_ai = ChessAI()
            self.chess_ai.load_model(movePredictor_path)

        self.layout = BoxLayout(orientation='horizontal', size_hint_y=0.9)
        self.add_widget(self.layout)

        self.image_widget = BGWidget(size_hint_x=0.7)
        self.layout.add_widget(self.image_widget)

        control_layout = BoxLayout(orientation='vertical', size_hint_x=0.3)
        self.layout.add_widget(control_layout)

        load_button = Button(text='Load Chessboard Image', font_size=16, font_name='Roboto')
        load_button.color = (1, 1, 1, 1)
        load_button.background_color = (0.5, 0.2, 0, 1)
        load_button.bind(on_release=self.load_image)
        control_layout.add_widget(load_button)

        detect_button = Button(text='Detect Pieces', font_size=16, font_name='Roboto')
        detect_button.color = (1, 1, 1, 1)
        detect_button.background_color = (0.5, 0.1, 0.3, 1)
        detect_button.bind(on_release=self.detect_pieces)
        control_layout.add_widget(detect_button)

        if CHESS_AI_AVAILABLE:
            predict_button = Button(text='Suggest Move', font_size=16, font_name='Roboto')
            predict_button.bind(on_release=self.turnSelect)
            predict_button.color = (1, 1, 1, 1)
            predict_button.background_color = (0, 0.8, 0.4, 1)
            control_layout.add_widget(predict_button)

        self.detected_board_dict = None  # Store detected board as dictionary
        self.current_turn = None  # Store whose turn it is

    def load_image(self, instance):
        filechooser = FileChooserListView()
        filechooser.filters = ['*.png', '*.jpg', '*.jpeg']
        popup = Popup(title='Select Image', content=filechooser, size_hint=(0.9, 0.9))

        def on_selection(instance, selection):
            if selection:
                self.display_image(selection[0])
                popup.dismiss()

        filechooser.bind(selection=on_selection)
        popup.open()

    def display_image(self, image_path):
        try:
            self.image_widget.set_source(image_path)
            self.current_image_path = image_path
        except Exception as e:
            Popup(title='Error', content=Label(text=f'Failed to load image: {e}', font_name='Roboto'),
                  size_hint=(0.6, 0.4)).open()

    def detect_pieces(self, instance):
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            Popup(title='Error', content=Label(text='Please load an image first!', font_name='Roboto'),
                  size_hint=(0.6, 0.4)).open()
            return

        try:
            detections, image = self.board_detector.detect_pieces(self.current_image_path)
            board_squares, _ = self.board_detector.squareEstimates(image.shape, detections)

            # Convert to board dict
            files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
            board_dict = {}

            for (row, col), pieces in board_squares.items():
                square = f"{files[col]}{ranks[row]}"
                board_dict[square] = pieces[0]['class']
            self.detected_board_dict = board_dict

            Popup(title='Success', content=Label(text='Pieces detected', font_size=16, font_name='Roboto'),
                  size_hint=(0.6, 0.4)).open()

        except Exception as e:
            Popup(title='Error', content=Label(text=f'Detection failed: {e}', font_name='Roboto'),
                  size_hint=(0.6, 0.4)).open()
            print(f"Detection error: {e}")

    def turnSelect(self, instance):
        if not CHESS_AI_AVAILABLE:
            Popup(title='Error', content=Label(text='Chess AI not available!', font_name='Roboto'),
                  size_hint=(0.6, 0.4)).open()
            return

        if not self.detected_board_dict:
            Popup(title='Error', content=Label(text='Please detect pieces first!', font_name='Roboto'),
                  size_hint=(0.6, 0.4)).open()
            return

        # Show turn selection popup
        turn_popup = TurnSelectionPopup(self.predictMove)
        turn_popup.open()

    def predictMove(self, turn):
        self.current_turn = turn
        
        try:
            # Get all predictions from the model
            all_predictions = self.chess_ai.predictBestMove(self.detected_board_dict, top_k=20)
            
            if not all_predictions:
                Popup(
                    title='Result', 
                    content=Label(text='No move could be suggested.', font_name='Roboto'),
                    size_hint=(0.6, 0.4)
                ).open()
                return
            
            # Filter predictions to only include valid moves for the current turn
            valid_moves = []
            for move, confidence in all_predictions:
                if self.isValid(move, turn, self.detected_board_dict):
                    valid_moves.append((move, confidence))
                
                if len(valid_moves) >= 5:
                    break
            
            if valid_moves:
                # Get the best valid move
                best_move, best_confidence = valid_moves[0]
                turn_text = "White" if turn == 'white' else "Black"
                
                popup_content = BoxLayout(orientation='vertical', padding=20, spacing=10)
                
                turn_label = Label(
                    text=f"Turn: {turn_text}",
                    font_size=18,
                    font_name='Roboto',
                    size_hint_y=0.2
                )
                popup_content.add_widget(turn_label)
                
                move_label = Label(
                    text=f"Suggested Move: {best_move}",
                    font_size=20,
                    font_name='Roboto',
                    size_hint_y=0.3
                )
                popup_content.add_widget(move_label)
                
                # Convert confidence to percentage
                conf_percentage = best_confidence * 100
                conf_label = Label(
                    text=f"Confidence: {conf_percentage:.2f}%",
                    font_size=16,
                    font_name='Roboto',
                    size_hint_y=0.2
                )
                popup_content.add_widget(conf_label)
                
                # Show alternative moves if available
                if len(valid_moves) > 1:
                    alt_text = "Alternatives: "
                    alt_moves = [f"{move} ({conf*100:.1f}%)" for move, conf in valid_moves[1:3]]
                    alt_text += ", ".join(alt_moves)
                    
                    alt_label = Label(
                        text=alt_text,
                        font_size=12,
                        font_name='Roboto',
                        size_hint_y=0.3,
                        text_size=(None, None),
                        halign='center'
                    )
                    popup_content.add_widget(alt_label)
                
                Popup(
                    title='Move Suggestion',
                    content=popup_content,
                    size_hint=(0.8, 0.6)
                ).open()
            else:
                # No valid moves found for this turn
                turn_text = "White" if turn == 'white' else "Black"
                Popup(
                    title='No Valid Moves', 
                    content=Label(
                        text=f'No valid moves found for {turn_text}.\nPlease check the board detection.',
                        font_name='Roboto',
                        halign='center'
                    ),
                    size_hint=(0.7, 0.4)
                ).open()
                
        except Exception as e:
            Popup(
                title='Error', 
                content=Label(text=f'Move prediction failed: {e}', font_name='Roboto'),
                size_hint=(0.8, 0.4)
            ).open()
            print(f"Prediction error: {e}")
    
    def setupBoard(self, board_dict, turn):
        return board_dict.copy()
    
    def isValid(self, move, turn, board_dict):
        try:
            # Parse the move
            if len(move) < 4:
                return False
                
            from_square = move[:2].lower()
            to_square = move[2:4].lower()
            
            # Check if there's a piece on the from_square
            if from_square not in board_dict:
                return False
            
            piece = board_dict[from_square]
            
            # Skip empty squares (marked as 'board')
            if piece == 'board':
                return False
            
            # Check if the piece belongs to the current player
            if turn == 'white':
                return piece.startswith('White')
            elif turn == 'black':
                return piece.startswith('Black')
            
            return False
            
        except Exception as e:
            print(f"Error validating move {move}: {e}")
            return False

class GlobchessApp(App):
    def build(self):
        return Globchess()

if __name__ == '__main__':
    GlobchessApp().run()