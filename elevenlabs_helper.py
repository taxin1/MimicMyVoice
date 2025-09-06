import requests

API_KEY = 'sk_1f48b5e0e6db51a46f56458c8dc395c73aaee5d7fea998f4'
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'  # example voice ID; you can use the default or your custom voice ID

def elevenlabs_synthesize(text: str, output_path: str):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/wav"
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    else:
        raise Exception(f"ElevenLabs API error: {response.status_code} {response.text}")
