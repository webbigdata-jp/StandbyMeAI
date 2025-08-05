import asyncio
import httpx
import json
import re
import torch
from snac import SNAC
import scipy.io.wavfile as wavfile
import numpy as np
import pyaudio
import threading
import time
import subprocess
import sys
from transformers import AutoTokenizer

class VoiceCoreSystem:
    def __init__(self, 
                 gemma_server_url="http://localhost:8080",
                 voicecore_server_url="http://localhost:8081"):
        self.gemma_server_url = gemma_server_url
        self.voicecore_server_url = voicecore_server_url
        self.snac_model = None
        self.tokenizer = None
        self.load_models()
    
    def load_models(self):
        """SNACモデルとトークナイザーを読み込み"""
        print("SNACモデルを読み込んでいます...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model.to("cpu")
        
        print("Gemmaトークナイザーを読み込んでいます...")
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E2B-it")
    
    async def chat_with_gemma(self, user_input):
        """Gemma 3nとチャット"""
        # Gemma 3nのチャット形式（手動でテンプレートを作成）
        chat_prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        # Gemmaサーバーにリクエスト
        payload = {
            "prompt": chat_prompt,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.00,
            "repeat_penalty": 1.0,
            "n_predict": 512,
            "stop": ["<end_of_turn>", "<|end_of_text|>"]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.gemma_server_url}/completion",
                    json=payload
                )
                
                print(f"Status Code: {response.status_code}")
                print(f"Response Headers: {response.headers}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Full response: {result}")
                    content = result.get('content', '').strip()
                    return content
                else:
                    print(f"Gemmaサーバーエラー: {response.status_code}")
                    print(f"Response text: {response.text}")
                    return None
        except Exception as e:
            print(f"リクエストエラー: {e}")
            return None
    
    async def generate_voice_tokens(self, text):
        """VoiceCoreで音声トークンを生成"""
        # VoiceCore用のプロンプト形式
        voice_prompt = f"<custom_token_3><|begin_of_text|>matsukaze_male[neutral]: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
        
        payload = {
            "prompt": voice_prompt,
            "temperature": 0.8,
            "top_p": 0.95,
            "n_predict": 2048,
            "repeat_penalty": 1.1,
            "repeat_last_n": 70
        }
        
        collected_tokens = []
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    f"{self.voicecore_server_url}/completion",
                    json=payload,
                    headers={"Accept": "application/x-ndjson"}
                )
                
                async for line in response.aiter_text():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "content" in data:
                                matches = re.findall(r'<custom_token_(\d+)>', data["content"])
                                for match in matches:
                                    token_id = 128256 + int(match)
                                    collected_tokens.append(token_id)
                        except:
                            pass
            
            return collected_tokens
        except Exception as e:
            print(f"VoiceCoreエラー: {e}")
            return []
    
    def redistribute_codes(self, tokens):
        """トークンをSNACコード形式に変換"""
        code_list = [t - 128266 for t in tokens]
        layer_1, layer_2, layer_3 = [], [], []
        
        for i in range(len(code_list) // 7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        
        return [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
    
    def snac_decode_to_audio(self, tokens):
        """SNACで音声データに変換"""
        if not tokens:
            return None
        
        # 7の倍数に調整
        code_length = (len(tokens) // 7) * 7
        if code_length == 0:
            return None
        
        tokens = tokens[:code_length]
        codes = self.redistribute_codes(tokens)
        
        # 音声デコード
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
            audio_np = audio_hat.detach().squeeze().cpu().numpy()
        
        return audio_np
    
    def play_audio(self, audio_data, sample_rate=24000):
        """音声を再生"""
        if audio_data is None:
            return
        
        # PyAudioで再生
        p = pyaudio.PyAudio()
        
        stream = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=sample_rate,
                       output=True)
        
        # numpy配列をbytesに変換
        audio_bytes = audio_data.astype(np.float32).tobytes()
        stream.write(audio_bytes)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def save_audio(self, audio_data, filename, sample_rate=24000):
        """音声をファイルに保存"""
        if audio_data is not None:
            wavfile.write(filename, sample_rate, audio_data)
            print(f"音声を{filename}に保存しました")
    
    async def process_conversation(self, user_input, save_file=None, play_audio=True):
        """会話全体の処理"""
        print(f"\nユーザー: {user_input}")
        
        # Step 1: Gemma 3nで回答生成
        print("Gemma 3nで回答を生成中...")
        gemma_response = await self.chat_with_gemma(user_input)
        
        if not gemma_response:
            print("Gemmaからの回答が取得できませんでした")
            return
        
        print(f"Gemma回答: {gemma_response}")
        
        # Step 2: VoiceCoreで音声トークン生成
        print("VoiceCoreで音声トークンを生成中...")
        voice_tokens = await self.generate_voice_tokens(gemma_response)
        
        if not voice_tokens:
            print("音声トークンの生成に失敗しました")
            return
        
        print(f"音声トークン数: {len(voice_tokens)}")
        
        # Step 3: SNACで音声化
        print("SNACで音声化中...")
        audio_data = self.snac_decode_to_audio(voice_tokens)
        
        if audio_data is None:
            print("音声の生成に失敗しました")
            return
        
        # 保存（オプション）
        if save_file:
            self.save_audio(audio_data, save_file)
        
        # 再生（オプション）
        if play_audio:
            print("音声を再生中...")
            self.play_audio(audio_data)
        
        return audio_data

# 使用例
async def main():
    # システムを初期化
    voice_system = VoiceCoreSystem()
    
    print("VoiceCore統合システムが起動しました！")
    print("使用方法:")
    print("1. Gemmaサーバーを起動: ポート8080")
    print("2. VoiceCoreサーバーを起動: ポート8081")
    print("3. 'quit'で終了")
    
    while True:
        try:
            user_input = input("\n質問を入力してください: ")
            
            if user_input.lower() == 'quit':
                break
            
            # 音声ファイル名（オプション）
            timestamp = int(time.time())
            save_file = f"output_{timestamp}.wav"
            
            # 会話処理
            await voice_system.process_conversation(
                user_input, 
                save_file=save_file, 
                play_audio=True
            )
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
    
    print("システムを終了します")

if __name__ == "__main__":
    asyncio.run(main())