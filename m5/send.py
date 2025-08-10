import asyncio
import httpx
import json
import re
import torch
from snac import SNAC
import scipy.io.wavfile as wavfile
import numpy as np

async def generate_voice(prompt, output_file="output.wav"):
    """シンプルな音声生成関数"""
    
    # SNACモデルの読み込み
    print("SNACモデルを読み込んでいます...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model.to("cpu")
    
    # サーバーにリクエスト送信
    payload = {
        "prompt": prompt,
        "temperature": 0.8,
        "top_p": 0.95,
        "n_predict": 2048,
        "repeat_penalty": 1.1,
        "repeat_last_n": 70
    }
    
    collected_tokens = []
    
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            "http://localhost:8080/completion",
            json=payload,
            headers={"Accept": "application/x-ndjson"}
        )
        
        # トークンの収集
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
    
    # 音声の生成
    if collected_tokens:
        # 7の倍数に調整
        code_length = (len(collected_tokens) // 7) * 7
        tokens = collected_tokens[:code_length]
        
        # コードの再分配
        codes = redistribute_codes(tokens)
        
        # 音声デコード
        with torch.inference_mode():
            audio_hat = snac_model.decode(codes)
        
        audio_np = audio_hat.detach().squeeze().cpu().numpy()
        
        # WAVファイルとして保存
        wavfile.write(output_file, 24000, audio_np)
        print(f"音声を {output_file} に保存しました。")

def redistribute_codes(tokens):
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

# 使用例
async def main():
    # 松風さんの声で挨拶
    prompt = "<custom_token_3><|begin_of_text|>amitaro_female[neutral]: こんにちは！よろしくね！<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    await generate_voice(prompt, "greeting1.wav")
    
    prompt = "<custom_token_3><|begin_of_text|>amitaro_female[neutral]: もう、子ども扱いしないでね！ぶー<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    await generate_voice(prompt, "greeting2.wav")

    prompt = "<custom_token_3><|begin_of_text|>amitaro_female[neutral]: まぁ、許してあげる！うふふ<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    await generate_voice(prompt, "greeting3.wav")


if __name__ == "__main__":
    asyncio.run(main())
