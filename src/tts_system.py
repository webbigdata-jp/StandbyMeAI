import httpx
import re
import torch
import time
import numpy as np
import pyaudio
import scipy.io.wavfile as wavfile
from transformers import AutoTokenizer
from snac import SNAC

# --- Debug Settings ---
DEBUG = True

# -----------------------------------------------------------------------------
# Class to generate audio from a text message
# -----------------------------------------------------------------------------
class TextToSpeechSystem:
    def __init__(self, voicecore_server_url="http://localhost:8081"):
        self.voicecore_server_url = voicecore_server_url
        self.snac_model = None
        self.text_tokenizer = None
        self._load_models()

    def _load_models(self):
        print("\n=== Loading Speech Synthesis Models ===")
        try:
            tokenizer_name = "webbigdata/VoiceCore"
            self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print("  ✓ Tokenizer loaded successfully")
            
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
            print("  ✓ SNAC model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading speech synthesis models: {e}")
            raise

    async def _generate_voice_tokens(self, text):
        print("  [1] Converting prompt to token IDs and sending request to VoiceCore server...")
        ids = {"soh": 128259, "bot": 128000, "eot": 128009, "eoh": 128260, "soai": 128261, "sos": 128257}
        text_ids = self.text_tokenizer(f"matsukaze_male[neutral]: {text}", add_special_tokens=False).input_ids
        prompt_ids = [ids["soh"], ids["bot"]] + text_ids + [ids["eot"], ids["eoh"], ids["soai"], ids["sos"]]
        payload = {"prompt": prompt_ids, "temperature": 0.8, "top_p": 0.95, "n_predict": 4096}
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(f"{self.voicecore_server_url}/completion", json=payload)
            if response.status_code != 200:
                print(f"✗ VoiceCore server error: {response.status_code}\n{response.text}")
                return []
            
            matches = re.findall(r'<custom_token_(\d+)>', response.json().get("content", ""))
            return [128256 + int(m) for m in matches] if matches else []
        except Exception as e:
            print(f"✗ VoiceCore token generation error: {e}")
            return []

    def _redistribute_codes(self, tokens):
        """Redistributes a flat list of tokens into three layers for the SNAC model."""
        code_list = [t - 128266 for t in tokens]
        layer_1, layer_2, layer_3 = [], [], []
        num_chunks = len(code_list) // 7
        
        for i in range(num_chunks):
            base = 7 * i
            layer_1.append(code_list[base])
            layer_2.append(code_list[base + 1] - 4096)
            layer_3.append(code_list[base + 2] - (2 * 4096))
            layer_3.append(code_list[base + 3] - (3 * 4096))
            layer_2.append(code_list[base + 4] - (4 * 4096))
            layer_3.append(code_list[base + 5] - (5 * 4096))
            layer_3.append(code_list[base + 6] - (6 * 4096))
            
        return [torch.tensor(layer).unsqueeze(0) for layer in (layer_1, layer_2, layer_3)]

    def _snac_decode_to_audio(self, tokens):
        if not tokens or len(tokens) < 7:
            return None
        codes = self._redistribute_codes(tokens)
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
        return audio_hat.detach().squeeze().cpu().numpy()

    def save_audio(self, audio_data, filename, sample_rate=24000):
        if audio_data is not None:
            if DEBUG: print(f"  [3] Saving audio data to file: {filename}")
            wavfile.write(filename, sample_rate, audio_data.astype(np.float32))

    def play_audio(self, audio_data, sample_rate=24000):
        if audio_data is None:
            return
        if DEBUG: print("  [4] Playing audio...")
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
            stream.write(audio_data.astype(np.float32).tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
            if DEBUG: print("  ✓ Playback complete.")
        except Exception as e:
            print(f"  ✗ Audio playback error: {e}")

    async def generate_speech_data(self, text, save_filename):
        print(f"\n--- Starting speech synthesis ---\nText: \"{text}\"")
        voice_tokens = await self._generate_voice_tokens(text)
        if not voice_tokens:
            print("✗ Failed to get voice tokens")
            return None, 0
        
        audio_data = self._snac_decode_to_audio(voice_tokens)
        if audio_data is None:
            print("✗ Failed to generate audio data")
            return None, 0
        
        self.save_audio(audio_data, save_filename)
        duration = audio_data.size / 24000
        print(f"--- Speech synthesis finished (Generated {duration:.2f}s of audio) ---\n")
        return audio_data, duration
