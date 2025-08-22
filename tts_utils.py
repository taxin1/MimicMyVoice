import tempfile
import os
import soundfile as sf
import pyttsx3

def text_to_speech(text, rate=200, gender=None):
    """
    Convert text to speech using pyttsx3 (system voices) and save as wav file
    
    Args:
        text (str): Text to convert to speech
        rate (int): Speaking rate (default: 200)
        gender (str): Gender ('male' or 'female')
    
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Initialize TTS engine
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        # Check if we have any voices
        if not voices:
            raise Exception("No TTS voices found on your system")
            
        # We know from our scan that:
        # voices[0] is Microsoft David (male)
        # voices[1] is Microsoft Zira (female)
        
        # Set voice based on gender
        if gender and gender.lower() == 'female':
            # Use voice[1] for female (Microsoft Zira)
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            else:
                # Fallback if there's only one voice
                engine.setProperty('voice', voices[0].id)
        else:
            # Use voice[0] for male (Microsoft David)
            engine.setProperty('voice', voices[0].id)
        
        # Set rate
        engine.setProperty('rate', rate)
        
        # Create a temporary WAV file
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        
        # Generate and save speech
        engine.save_to_file(text, tmp_wav.name)
        engine.runAndWait()
            
        return tmp_wav.name
    
    except Exception as e:
        raise Exception(f"TTS generation failed: {str(e)}")

def get_voice_names():
    """
    Get names of available voices
    
    Returns:
        tuple: (male_name, female_name) - names of male and female voices
    """
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        male_name = "Male Voice (David)"
        female_name = "Female Voice (Zira)"
        
        # Get actual names if available
        if len(voices) > 0:
            male_name = voices[0].name
        if len(voices) > 1:
            female_name = voices[1].name
            
        return (male_name, female_name)
        
    except Exception as e:
        print(f"Error getting voices: {e}")
        return ("Male Voice", "Female Voice")
