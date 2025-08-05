import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
from PIL import Image

resultPositions = {}

class ChessboardDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.piece_colors = {
            'white': '#FFFFFF',
            'black': '#000000',
            'unknown': '#FF0000'
        }
        
    def detect_pieces(self, image_path, confidence_threshold=0.5):
        try:
            pil_image = Image.open(image_path)
            image_rgb = np.array(pil_image)
            
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_rgb
                
        except Exception as e:
            # Fallback to OpenCV
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image from {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = self.model(image_path, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name,
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    })
        
        return detections, image_rgb
    
    def squareEstimates(self, image_shape, detections):
        height, width = image_shape[:2]
        
        if detections:
            all_centers = [d['center'] for d in detections]
            min_x = min(center[0] for center in all_centers)
            max_x = max(center[0] for center in all_centers)
            min_y = min(center[1] for center in all_centers)
            max_y = max(center[1] for center in all_centers)
            
            padding_x = (max_x - min_x) * 0.1
            padding_y = (max_y - min_y) * 0.1
            
            board_left = max(0, min_x - padding_x)
            board_right = min(width, max_x + padding_x)
            board_top = max(0, min_y - padding_y)
            board_bottom = min(height, max_y + padding_y)
        else:
            # Use the entire image if no detections
            board_left, board_top = 0, 0
            board_right, board_bottom = width, height
        
        # Calculate square dimensions
        square_width = (board_right - board_left) / 8
        square_height = (board_bottom - board_top) / 8
        
        # Map each detection to a square
        board_squares = {}
        
        for detection in detections:
            center_x, center_y = detection['center']
            
            # Calculate which square this piece is in
            col = int((center_x - board_left) / square_width)
            row = int((center_y - board_top) / square_height)
            
            col = max(0, min(7, col))
            row = max(0, min(7, row))
            
            # Store the piece information
            square_key = (row, col)
            if square_key not in board_squares:
                board_squares[square_key] = []
            
            board_squares[square_key].append(detection)
        
        return board_squares, (board_left, board_top, board_right, board_bottom)
    
    def drawBoard(self, image, detections, save_path=None, show_display=False):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        ax.imshow(image)
        
        # Get board squares
        board_squares, board_bounds = self.squareEstimates(image.shape, detections)
        board_left, board_top, board_right, board_bottom = board_bounds
        
        # Calculate square dimensions
        square_width = (board_right - board_left) / 8
        square_height = (board_bottom - board_top) / 8
        
        # Draw the 8x8 grid
        for row in range(9):  # 9 lines for 8 squares
            y = board_top + row * square_height
            ax.axhline(y=y, color='red', linewidth=2, alpha=0.7)
        
        for col in range(9):  # 9 lines for 8 squares
            x = board_left + col * square_width
            ax.axvline(x=x, color='red', linewidth=2, alpha=0.7)
        
        # Draw piece detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='lime', 
                           facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            ax.text(x1, y1-5, label, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='lime', alpha=0.8),
                   color='black', weight='bold')
        
        # Add square labels in chess notation (files and ranks)
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        for row in range(8):
            for col in range(8):
                center_x = board_left + (col + 0.5) * square_width
                center_y = board_top + (row + 0.5) * square_height
                
                square_name = f"{files[col]}{ranks[row]}"
                
                # Check if there's a piece in this square
                if (row, col) in board_squares:
                    # pieces = board_squares[(row, col)]
                    piece_info = f"{square_name}"
                    ax.text(center_x, center_y, piece_info, 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='yellow', alpha=0.7),
                           weight='bold')
                else:
                    ax.text(center_x, center_y, square_name, 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='white', alpha=0.5))
        
        # Set title and labels
        ax.set_title(f'Chess Piece Detection - {len(detections)} pieces detected', 
                    fontsize=16, weight='bold')
        ax.set_xlabel('Chessboard with detected pieces and square locations', fontsize=12)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result saved to: {save_path}")
        else:
            plt.savefig("chess_detection_result.png", dpi=300, bbox_inches='tight')
            print("Result saved to: chess_detection_result.png")
        
        if show_display:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display image: {e}")
                print("Running in headless environment. Image saved to file instead.")
        
        plt.close(fig)
        
        return fig
    
    def print_detection_summary(self, detections, board_squares):
        print(f"\n=== Detection Summary ===")
        print(f"Total pieces detected: {len(detections)}")
        print(f"Occupied squares: {len(board_squares)}")
        
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        print("\nPiece locations:")
        for (row, col), pieces in board_squares.items():
            square_name = f"{files[col]}{ranks[row]}"
            for piece in pieces:
                print(f"  {square_name}: {piece['class']} (confidence: {piece['confidence']:.2f})")
                resultPositions[square_name] = piece['class']
        
def get_result_positions():
    return resultPositions

def main():
    # Configuration
    model_path = "/home/nikhil/Desktop/ChessBot/chessPieceDetector.pt"
    image_path = "/home/nikhil/Desktop/ChessBot/test.png"
    output_path = "chess_detection_result.png"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please update the model_path variable with the correct path to your YOLOv8n model.")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please update the image_path variable with the correct path to your chessboard image.")
        return
    
    try:
        # Check for GUI environment
        import matplotlib
        gui_available = False
        try:
            # Try to use a GUI backend
            matplotlib.use('TkAgg')
            gui_available = True
        except:
            # Fall back to Agg backend for headless environments
            matplotlib.use('Agg')
            print("GUI not available, using headless mode. Image will be saved to file.")
        
        # Initialize detector
        detector = ChessboardDetector(model_path)
        
        # Detect pieces
        print("Detecting chess pieces...")
        detections, original_image = detector.detect_pieces(image_path, confidence_threshold=0.3)
        
        # Get board squares
        board_squares, _ = detector.squareEstimates(original_image.shape, detections)
        
        # Print summary
        detector.print_detection_summary(detections, board_squares)
        
        # Draw and display result
        print("\nGenerating visualization...")
        detector.drawBoard(
            original_image, 
            detections, 
            show_display=gui_available,
            save_path=output_path
        )
        
        print("\nVisualization complete. Check the saved image file.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required dependencies installed")

if __name__ == "__main__":
    main()