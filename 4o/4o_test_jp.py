import os
import json
from openai import OpenAI

# 1) Set API key in environment variables
os.environ["OPENAI_API_KEY"] = "********************************************************"

# 2) Initialize client
client = OpenAI()

# 3) List of inference files and prefixes
inference_files = [
    ("adversarial_jp", "full_adversarial_jp.json")
]

# 4) Create log directory
os.makedirs("logs_jp", exist_ok=True)

# 5) Number of runs
RUNS_PER_MODEL = 10

# 6) Process each file
for prefix, filepath in inference_files:
    with open(filepath, "r", encoding="utf-8") as f:
        entries = json.load(f)
    total = len(entries)
    print(f"\n▶ Processing file '{filepath}' ({total} entries)")

    for idx, entry in enumerate(entries, start=1):
        print(f"  • Entry {idx}/{total}")
        messages = entry["input"]["messages"]

        for model_id, label in [
            ("gpt-4o-2024-08-06",                 "4o"),
            ("ft:gpt-4o-2024-08-06:[REDACTED]:kohlberg02-dpo:AsihZFA5", "4o_finetuned")
        ]:
            print(f"    ↳ [{label}] 10 runs on model '{model_id}'")
            for run_idx in range(1, RUNS_PER_MODEL + 1):
                print(f"        ◦ Run {run_idx}/{RUNS_PER_MODEL} ...", end="", flush=True)

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

                # Generate filename (with run index)
                fn = f"logs_jp/{prefix}_{idx:03d}_{label}_{run_idx:02d}.txt"
                with open(fn, "w", encoding="utf-8") as out:
                    out.write("=== Prompt ===\n")
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "").replace("\n", " ")
                        out.write(f"{role}: {content}\n")
                    out.write("\n=== Response ===\n")
                    out.write(result_text)

                print(" done, saved to", fn)

print("\n✅ All done!")
