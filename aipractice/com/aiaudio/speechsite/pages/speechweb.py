'''
Created on May 16, 2024

@author: ctatlah
'''

from flask import Flask, request, render_template, jsonify
from google.cloud import speech_v2 as speech

app = Flask(__name__, template_folder='../pages/templates')

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/speechin", methods=["POST"])
def handle_audio():
    # Get the audio data from the request
    audio_data = request.files.get("audio_data")
    
    if audio_data:
        # Get text of speech audio
        text = transcribeAudio(audio_data)
        
        return jsonify({"message": text}), 200
    else:
        return jsonify({'error': 'No audio data found'}), 400

def transcribeAudio(audio_data):
    """
    Transcribe the audio of given data using Google Cloud Speech
    Args:
      audio : audio media
    Returns:
      text : transcription of the audio
      errorMsg : any error messages 
    """
    # Setup Google Cloud Speech objects
    client = speech.SpeechClient.from_service_account_json("../../service_account.json")
    audio_content = audio_data.read()
    config = speech.RecognitionConfig(
          encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
          sample_rate_hertz=16000,
          language_code="en-US",
    )
    
    # Transcribe
    audio = speech.Audio(content=audio_content)
    response = client.recognize(config=config, audio=audio)
    text = response.results[0].alternatives[0].transcript
        
    return text

if __name__ == "__main__":
    app.run(debug=True)