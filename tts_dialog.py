from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from tts_utils import text_to_speech

class TTSDialog(QDialog):
    tts_generated = pyqtSignal(str)  # Signal to emit when TTS is generated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text-to-Speech Generator")
        self.setMinimumSize(500, 300)
        self.tts_path = None
        self.setup_ui()
        self.load_available_voices()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Text input
        layout.addWidget(QLabel("Enter text to synthesize:"))
        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)
        
        # Gender selection
        gender_layout = QHBoxLayout()
        gender_layout.addWidget(QLabel("Voice Gender:"))
        self.gender_combo = QComboBox()
        self.gender_combo.addItem("Male")
        self.gender_combo.addItem("Female")
        gender_layout.addWidget(self.gender_combo)
        layout.addLayout(gender_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("Slow", 150)
        self.speed_combo.addItem("Normal", 200)
        self.speed_combo.addItem("Fast", 250)
        speed_layout.addWidget(self.speed_combo)
        layout.addLayout(speed_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.clicked.connect(self.generate_speech)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def load_available_voices(self):
        """No need to load voices anymore, just a placeholder for compatibility"""
        # No need to do anything as we're only using gender selection now
        pass
    
    def generate_speech(self):
        """Generate TTS from the entered text"""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Text", "Please enter some text to synthesize.")
            return
            
        # Get selected gender
        selected_gender = self.gender_combo.currentText().lower()
            
        rate = self.speed_combo.currentData()  # Get selected speed value
        
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Generating speech... Please wait.")
        
        try:
            # Generate speech with selected gender and rate
            tts_path = text_to_speech(text, rate, selected_gender)
            self.tts_path = tts_path
            self.tts_generated.emit(tts_path)
            self.accept()  # Close dialog when done
            
        except Exception as e:
            QMessageBox.critical(self, "TTS Generation Error", str(e))
            self.status_label.setText("Error generating speech.")
            
        finally:
            self.generate_btn.setEnabled(True)
