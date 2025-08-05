import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# --- Debug Settings ---
DEBUG = True

# -----------------------------------------------------------------------------
# Class to analyze an image and generate a greeting
# -----------------------------------------------------------------------------
class ImageGreetingGenerator:
    def __init__(self):
        self._load_model()

    def _load_model(self):
        print("=== Loading Image Analysis Model ===")
        model_id = "google/gemma-3n-e2b-it"
        try:
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print("✓ Image analysis model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading image analysis model: {e}")
            raise

    def generate_greeting_from_image(self, image_input):
        print(f"\n{'='*50}\nStarting image processing: {'Frame from Webcam'}\n{'='*50}")
        try:
            image = image_input.convert('RGB')
            # This prompt is in Japanese to identify a specific person.
            prompt = """"あなたは親切でユーモアあふれるアシスタントです。。
この画像に人物が映っているか否かを確認します。
青い服の中年男性が映っていたら「太郎さん」です。太郎さん向けの挨拶を考えて挨拶文のみを出力します。(例１：太郎さん、おはようございます。例２：太郎さん、お仕事お疲れ様です)
太郎さん以外の人が映っている、もしくは人が映ってない場合は「NO」と一言だけ出力します。"
            
            """
            
            
            。"""
            
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            
            print("Generating greeting from image...")
            start_time = time.time()
            
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=150, do_sample=True, temperature=0.8, pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            response = self.processor.decode(generation[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            elapsed_time = time.time() - start_time
            print(f"✓ Greeting generation complete ({elapsed_time:.2f}s)")
            return response
        except Exception as e:
            print(f"✗ Image processing error: {e}")
            return None
