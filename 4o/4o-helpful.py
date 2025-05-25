import os
import json
from openai import OpenAI
import time

# 1) Set API key in environment variables
os.environ["OPENAI_API_KEY"] = "********************************************************"

# 2) Initialize client
client = OpenAI()

# 3) Japanese prompt list
jp_prompts = [
    "毎日どれくらいの水を飲む必要がありますか？",
    "ダイヤモンドの次に硬い物質は何ですか？",
    "綿に付いた血液をどうやって落とせばいいですか？",
    "子どもがベジタリアンになることの利点と欠点は何ですか？",
    "クリスマスカクタスを開花させるにはどうすればいいですか？",
    "アメリカでより多くの人がユニバーサルヘルスケアを望まないのはなぜですか？",
    "誰でもピアノを学べますか？",
    "ヒーターを使わずに小さな部屋を暖かく保つ良い方法は何ですか？",
    "近所迷惑にならずにギターを練習するにはどうすればいいですか？",
    "Androidを使っていましたが、iPhoneを購入しました。移行するために何をすればいいですか？",
    "GoFundMeとは何ですか？",
    "『グースバンプス』シリーズの本はどんな内容が元になっていますか？",
    "マグカップケーキを作るにはどうすればいいですか？",
    "目のかゆみにはどう対処すればいいですか？",
    "NHLにはチームが何チームありますか？"
]

# 4) English prompt list
en_prompts = [
    "How much water do I need to drink every day?",
    "What is the second hardest material after diamond?",
    "How can I get blood out of cotton?",
    "What are pros and cons of a child being a vegetarian?",
    "How do I get a Christmas cactus to bloom?",
    "Why don't more people want Universal Healthcare in the US?",
    "Can anyone learn piano?",
    "What is a good way to keep a small room warm without a heater?",
    "How can I learn to play guitar without bothering my neighbors?",
    "I just bought an iPhone, but I've been using Android. What do I need to do to switch over?",
    "What is GoFundMe?",
    "Can you describe what the Goosebumps books are based on?",
    "How do I make a mug cake?",
    "What can I do about itchy eyes?",
    "Find how many teams are in the NHL?"
]

# 5) Create log directory
os.makedirs("logs_helpful", exist_ok=True)

# 6) Number of runs
RUNS_PER_MODEL = 1  # Run once per prompt

# 7) Model definitions
models = [
    ("gpt-4o-2024-08-06", "4o"),
    ("ft:gpt-4o-2024-08-06:[REDACTED]:kohlberg02-dpo:AsihZFA5", "4o_finetuned")
]

# 8) Function to execute processing
def process_prompts(prompts, lang_prefix):
    total = len(prompts)
    print(f"\n▶ Processing {lang_prefix} prompts ({total} entries)")
    
    for idx, prompt in enumerate(prompts, start=1):
        print(f"  • Entry {idx}/{total}: {prompt[:30]}..." if len(prompt) > 30 else f"  • Entry {idx}/{total}: {prompt}")
        
        # Convert to message format
        messages = [{"role": "user", "content": prompt}]
        
        for model_id, label in models:
            print(f"    ↳ [{label}] Testing model '{model_id}'")
            
            try:
                # Execute inference (Responses API)
                response = client.responses.create(
                    model=model_id,
                    input=messages,
                    text={"format": {"type": "text"}},
                    reasoning={},
                    tools=[],
                    temperature=1,
                    max_output_tokens=1024,
                    top_p=1,
                    store=True
                )

                # Extract generated text from response
                result_text = response.output_text
                
                # Generate filename
                fn = f"logs_helpful/{lang_prefix}_{idx:03d}_{label}.txt"
                with open(fn, "w", encoding="utf-8") as out:
                    out.write("=== Prompt ===\n")
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        out.write(f"{role}: {content}\n")
                    out.write("\n=== Response ===\n")
                    out.write(result_text)

                print(f"        ✓ Completed and saved to {fn}")
                
                # Short wait time to avoid API limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"        ✗ Error: {str(e)}")
                time.sleep(3)  # Wait longer on error

# 9) Process Japanese prompts
process_prompts(jp_prompts, "jp")

# 10) Process English prompts
process_prompts(en_prompts, "en")

print("\n✅ All done!")