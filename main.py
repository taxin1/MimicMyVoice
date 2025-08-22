import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from main_window import VoiceConversionApp

def main():
    app = QApplication(sys.argv)
    try:
        import librosa
        import scipy
        import soundfile
        import matplotlib
    except ImportError as e:
        QMessageBox.critical(
            None,
            "Missing Dependency",
            f"Missing package: {e}\nPlease install required packages with:\n  pip install librosa scipy soundfile matplotlib pyttsx3",
        )
        sys.exit(1)
    window = VoiceConversionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
