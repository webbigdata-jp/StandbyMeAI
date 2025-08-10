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
import queue

class VoiceCoreSystem:
    def __init__(self, 
                 gemma_server_url="http://localhost:8080",
                 voicecore_server_url="http://localhost:8081",
                 system_prompt=None):
        self.gemma_server_url = gemma_server_url
        self.voicecore_server_url = voicecore_server_url
        self.snac_model = None
        self.tokenizer = None
        self.system_prompt = system_prompt or "あなたは親切で役立つアシスタントです。"
        self.conversation_history = []
        
        # ストリーミング音声用の設定
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        self.sample_rate = 24000
        
        self.load_models()
    
    def load_models(self):
        """SNACモデルとトークナイザーを読み込み"""
        print("SNACモデルを読み込んでいます...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model.to("cpu")
        
        print("Gemmaトークナイザーを読み込んでいます...")
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E2B-it")
    
    def format_chat_prompt(self, messages):
        """メッセージリストをGemma 3Nのチャットテンプレート形式に変換"""
        prompt = ""
        
        for i, message in enumerate(messages):
            is_last = (i == len(messages) - 1)
            role = message["role"]
            content = message["content"]
            
            if role in ["user", "system"]:
                prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                if is_last:
                    prompt += "<start_of_turn>model\n"
            
            elif role == "assistant":
                prompt += f"<start_of_turn>model\n{content}"
                if not is_last:
                    prompt += "<end_of_turn>\n"
        
        return prompt
    
    def set_system_prompt(self, system_prompt):
        """システムプロンプトを設定"""
        self.system_prompt = system_prompt
        self.conversation_history = []
        print(f"システムプロンプトを設定しました: {system_prompt}")
    
    async def chat_with_gemma(self, user_input, use_history=True):
        """Gemma 3Nとチャット"""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history)
        
        messages.append({"role": "user", "content": user_input})
        
        chat_prompt = self.format_chat_prompt(messages)
        
        print(f"\n=== 生成されたプロンプト ===\n{chat_prompt}\n=== プロンプト終了 ===\n")
        
        payload = {
            "prompt": chat_prompt,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.00,
            "repeat_penalty": 1.0,
            "n_predict": 512,
            "stop": ["<end_of_turn>", "<|end_of_text|>", "<start_of_turn>"]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.gemma_server_url}/completion",
                    json=payload
                )
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('content', '').strip()
                    
                    if use_history:
                        self.conversation_history.append({"role": "user", "content": user_input})
                        self.conversation_history.append({"role": "assistant", "content": content})
                        
                        max_history_length = 10
                        if len(self.conversation_history) > max_history_length * 2:
                            self.conversation_history = self.conversation_history[-(max_history_length * 2):]
                    
                    return content
                else:
                    print(f"Gemmaサーバーエラー: {response.status_code}")
                    print(f"Response text: {response.text}")
                    return None
        except Exception as e:
            print(f"リクエストエラー: {e}")
            return None
    
    def clear_conversation_history(self):
        """会話履歴をクリア"""
        self.conversation_history = []
        print("会話履歴をクリアしました")
    
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
        
        code_length = (len(tokens) // 7) * 7
        if code_length == 0:
            return None
        
        tokens = tokens[:code_length]
        codes = self.redistribute_codes(tokens)
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
            audio_np = audio_hat.detach().squeeze().cpu().numpy()
        
        return audio_np
    
    def start_playback_thread(self):
        """音声再生スレッドを開始"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
            print("🎵 音声再生スレッド開始")
    
    def stop_playback_thread(self):
        """音声再生スレッドを停止"""
        self.is_playing = False
        self.audio_queue.put(None)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
    
    def _playback_worker(self):
        """音声再生ワーカー（別スレッドで実行）"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=self.sample_rate,
                           output=True,
                           frames_per_buffer=2048)
            
            print("🎵 音声再生ワーカー開始 - PyAudio初期化完了")
            
            chunk_count = 0
            
            while self.is_playing:
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                    
                    if audio_data is None:
                        print("🛑 音声再生停止信号を受信")
                        break
                    
                    chunk_count += 1
                    print(f"🔊 音声チャンク#{chunk_count}再生開始: {len(audio_data)}サンプル")
                    
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                    
                    audio_bytes = audio_data.astype(np.float32).tobytes()
                    stream.write(audio_bytes)
                    print(f"✅ 音声チャンク#{chunk_count}再生完了")
                    
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    print("⏳ キュー待機中...")
                    continue
                except Exception as e:
                    print(f"❌ 音声再生エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            print(f"🎵 音声再生ワーカー終了 - 総再生チャンク数: {chunk_count}")
            
        except Exception as e:
            print(f"❌ 音声再生ワーカー初期化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    async def generate_voice_tokens_streaming(self, text, callback=None):
        """
        VoiceCoreで音声トークンをストリーミング生成
        70トークン毎にコールバックを呼び出し
        
        重要な修正: stream=true パラメータを追加し、実際のストリーミングレスポンスを受信
        """
        voice_prompt = f"<custom_token_3><|begin_of_text|>amitaro_female[neutral]: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
        
        payload = {
            "prompt": voice_prompt,
            "temperature": 0.8,
            "top_p": 0.95,
            "n_predict": 2048,
            "repeat_penalty": 1.1,
            "repeat_last_n": 70,
            "stream": True  # ★重要: ストリーミングを有効化
        }
        
        collected_tokens = []
        all_tokens = []
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                # ストリーミングレスポンスを受信
                async with client.stream(
                    'POST',
                    f"{self.voicecore_server_url}/completion",
                    json=payload,
                    headers={"Accept": "text/event-stream"}  # SSEまたはNDJSONに対応
                ) as response:
                    
                    print(f"📡 ストリーミング接続確立: {response.status_code}")
                    
                    # ストリーミングでデータを受信
                    async for line in response.aiter_lines():
                        if line.strip():
                            # "data: " プレフィックスを除去（SSEの場合）
                            if line.startswith("data: "):
                                line = line[6:]
                            
                            try:
                                data = json.loads(line)
                                
                                # トークンを抽出
                                if "content" in data:
                                    matches = re.findall(r'<custom_token_(\d+)>', data["content"])
                                    for match in matches:
                                        token_id = 128256 + int(match)
                                        collected_tokens.append(token_id)
                                        all_tokens.append(token_id)
                                        
                                        if len(collected_tokens) % 10 == 0:
                                            print(f"🎯 トークン受信中: {len(collected_tokens)}/70")
                                        
                                        # 70トークン毎にコールバック実行
                                        if len(collected_tokens) >= 70:
                                            processable_count = (len(collected_tokens) // 7) * 7
                                            if processable_count > 0:
                                                tokens_to_process = collected_tokens[:processable_count]
                                                remaining_tokens = collected_tokens[processable_count:]
                                                
                                                print(f"🎯 70トークン到達! 処理開始: {len(tokens_to_process)}トークン")
                                                if callback:
                                                    # コールバックを非同期で実行（ブロッキングを避ける）
                                                    await callback(tokens_to_process)
                                                
                                                collected_tokens = remaining_tokens
                                
                                # ストリーミング終了の判定
                                if data.get("stop", False):
                                    print("📍 ストリーミング終了信号を受信")
                                    break
                                    
                            except json.JSONDecodeError as e:
                                print(f"⚠️ JSON解析エラー（無視）: {e}")
                                continue
                            except Exception as e:
                                print(f"❌ トークン処理エラー: {e}")
                                continue
                
                # 残りのトークンも処理
                if collected_tokens:
                    processable_count = (len(collected_tokens) // 7) * 7
                    if processable_count > 0:
                        tokens_to_process = collected_tokens[:processable_count]
                        print(f"📦 最終バッチ処理: {len(tokens_to_process)}トークン")
                        if callback:
                            await callback(tokens_to_process)
            
            print(f"✅ ストリーミング完了 - 総トークン数: {len(all_tokens)}")
            return all_tokens
            
        except Exception as e:
            print(f"❌ VoiceCoreストリーミングエラー: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _streaming_audio_callback(self, tokens):
        """ストリーミング音声のコールバック関数"""
        try:
            print(f"🔄 音声変換開始: {len(tokens)}トークン")
            audio_data = self.snac_decode_to_audio(tokens)
            if audio_data is not None:
                print(f"✅ 音声変換完了: {len(audio_data)}サンプル")
                print(f"📨 キューサイズ（追加前）: {self.audio_queue.qsize()}")
                
                self.audio_queue.put(audio_data)
                print(f"📨 音声チャンクをキューに送信完了 - 新サイズ: {self.audio_queue.qsize()}")
                
                # キューが処理されるのを少し待機
                await asyncio.sleep(0.01)
            else:
                print(f"❌ 音声変換失敗: トークン数={len(tokens)}")
        except Exception as e:
            print(f"❌ 音声変換エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def save_audio(self, audio_data, filename, sample_rate=24000):
        """音声をファイルに保存"""
        if audio_data is not None:
            wavfile.write(filename, sample_rate, audio_data)
            print(f"音声を{filename}に保存しました")
    
    async def process_conversation_streaming(self, user_input, save_file=None, use_history=True):
        """ストリーミング音声付き会話処理"""
        print(f"\nユーザー: {user_input}")
        
        # Step 1: Gemma 3nで回答生成
        print("Gemma 3nで回答を生成中...")
        gemma_response = await self.chat_with_gemma(user_input, use_history=use_history)
        
        if not gemma_response:
            print("Gemmaからの回答が取得できませんでした")
            return
        
        print(f"Gemma回答: {gemma_response}")
        
        # Step 2: ストリーミング音声再生開始
        print("🎵 ストリーミング音声生成・再生を開始...")
        
        # 再生スレッドを開始
        self.start_playback_thread()
        
        try:
            # VoiceCoreでストリーミング音声トークン生成
            all_tokens = await self.generate_voice_tokens_streaming(
                gemma_response, 
                callback=self._streaming_audio_callback
            )
            
            print(f"📊 総音声トークン数: {len(all_tokens)}")
            print(f"⏱️  音声生成完了 - 再生継続中...")
            
            # 音声の再生が完了するまで待機
            estimated_duration = (len(all_tokens) // 70) * 1.0 + 2.0
            print(f"⏳ 推定再生時間: {estimated_duration:.1f}秒")
            
            start_wait = time.time()
            while not self.audio_queue.empty():
                await asyncio.sleep(0.1)
                if time.time() - start_wait > estimated_duration:
                    print("⚠️  再生タイムアウト")
                    break
            
            await asyncio.sleep(0.5)
            print("🎵 ストリーミング音声再生完了")
            
            # 保存（オプション）
            if save_file and all_tokens:
                print("💾 完全な音声ファイルを生成・保存中...")
                complete_audio = self.snac_decode_to_audio(all_tokens)
                if complete_audio is not None:
                    self.save_audio(complete_audio, save_file)
            
            return all_tokens
            
        except Exception as e:
            print(f"❌ ストリーミング処理エラー: {e}")
            return None
        finally:
            print("🛑 音声再生スレッドを停止中...")
            self.stop_playback_thread()
    
    async def process_conversation(self, user_input, save_file=None, play_audio=True, use_history=True):
        """従来の会話処理（非ストリーミング）"""
        print(f"\nユーザー: {user_input}")
        
        print("Gemma 3nで回答を生成中...")
        gemma_response = await self.chat_with_gemma(user_input, use_history=use_history)
        
        if not gemma_response:
            print("Gemmaからの回答が取得できませんでした")
            return
        
        print(f"Gemma回答: {gemma_response}")
        
        print("VoiceCoreで音声トークンを生成中...")
        voice_tokens = await self.generate_voice_tokens_streaming(gemma_response)
        
        if not voice_tokens:
            print("音声トークンの生成に失敗しました")
            return
        
        print(f"音声トークン数: {len(voice_tokens)}")
        
        print("SNACで音声化中...")
        audio_data = self.snac_decode_to_audio(voice_tokens)
        
        if audio_data is None:
            print("音声の生成に失敗しました")
            return
        
        if save_file:
            self.save_audio(audio_data, save_file)
        
        if play_audio:
            print("音声を再生中...")
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=self.sample_rate,
                           output=True)
            
            audio_bytes = audio_data.astype(np.float32).tobytes()
            stream.write(audio_bytes)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        return audio_data

# 使用例
async def main():
    custom_system_prompt = """あなたは親切なアシスタントです。ユーザーと音声でおしゃべりをしているので返答は完結に１文で返答します。絵文字や顔文字は使ってはいけません"""
    
    voice_system = VoiceCoreSystem(system_prompt=custom_system_prompt)
    
    print("VoiceCore統合システムが起動しました！")
    print("使用方法:")
    print("1. Gemmaサーバーを起動: ポート8080")
    print("2. VoiceCoreサーバーを起動: ポート8081")
    print("コマンド:")
    print("  'quit' - 終了")
    print("  'clear' - 会話履歴をクリア")
    print("  'system' - システムプロンプトを変更")
    print("  'nohistory' - 履歴なしで単発の質問")
    print("  'streaming' - ストリーミング音声モード")
    print("  'normal' - 通常音声モード")
    
    streaming_mode = True
    
    while True:
        try:
            mode_indicator = "[ストリーミング]" if streaming_mode else "[通常]"
            user_input = input(f"\n{mode_indicator}質問を入力してください: ")
            
            if user_input.lower() == 'quit':
                break
            
            elif user_input.lower() == 'clear':
                voice_system.clear_conversation_history()
                continue
            
            elif user_input.lower() == 'system':
                new_prompt = input("新しいシステムプロンプトを入力してください: ")
                voice_system.set_system_prompt(new_prompt)
                continue
            
            elif user_input.lower() == 'streaming':
                streaming_mode = True
                print("ストリーミングモードに切り替えました")
                continue
            
            elif user_input.lower() == 'normal':
                streaming_mode = False
                print("通常モードに切り替えました")
                continue
            
            elif user_input.lower().startswith('nohistory '):
                actual_input = user_input[10:]
                timestamp = int(time.time())
                save_file = f"output_{timestamp}.wav"
                
                if streaming_mode:
                    await voice_system.process_conversation_streaming(
                        actual_input, 
                        save_file=save_file,
                        use_history=False
                    )
                else:
                    await voice_system.process_conversation(
                        actual_input, 
                        save_file=save_file, 
                        play_audio=True,
                        use_history=False
                    )
            
            else:
                timestamp = int(time.time())
                save_file = f"output_{timestamp}.wav"
                
                if streaming_mode:
                    await voice_system.process_conversation_streaming(
                        user_input, 
                        save_file=save_file,
                        use_history=True
                    )
                else:
                    await voice_system.process_conversation(
                        user_input, 
                        save_file=save_file, 
                        play_audio=True,
                        use_history=True
                    )
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    print("システムを終了します")

if __name__ == "__main__":
    asyncio.run(main())