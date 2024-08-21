"""
This script (optional) can rewrite and beautify the user-uploaded prompt via LLMs, mapping it to the style of EasyAnimate's training captions,
making it more suitable as the inference prompt and thus improving the quality of the generated videos.

Usage:
+ You can request OpenAI compatible server to perform beautiful prompt by running
```shell
export OPENAI_API_KEY="your_openai_api_key" OPENAI_BASE_URL="your_openai_base_url" python beautiful_prompt.py \
    --model "your_model_name" \
    --prompt "your_prompt"
```
+ You can also deploy the OpenAI Compatible Server locally using vLLM. For example:
```shell
# Meta-Llama-3-8B-Instruct is sufficient for this task.
# Download it from https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct or https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct to /path/to/your_llm

# deploy the OpenAI compatible server
python -m vllm.entrypoints.openai.api_server serve /path/to/your_llm --dtype auto --api-key "your_api_key"
```

Then you can perform beautiful prompt by running
```shell
python -m beautiful_prompt.py \
    --model /path/to/your_llm \
    --prompt "your_prompt" \
    --base_url "http://localhost:8000/v1" \
    --api_key "your_api_key"
```
"""
import argparse
import os

from openai import OpenAI

from easyanimate.video_caption.caption_rewrite import extract_output


def parse_args():
    parser = argparse.ArgumentParser(description="Beautiful prompt.")
    parser.add_argument("--model", type=str, required=True, help="The OpenAI model or the path to your local LLM.")
    parser.add_argument("--prompt", type=str, required=True, help="The user-uploaded prompt.")
    parser.add_argument(
        "--template",
        type=str,
        default="easyanimate/video_caption/prompt/beautiful_prompt.txt",
        help="A string or a txt file contains the template for beautiful prompt."
    )
    parser.add_argument(
        "--max_retry_nums",
        type=int,
        default=5,
        help="Maximum number of retries to obtain an output that meets the JSON format."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI API server url. If it is None, the OPENAI_BASE_URL from the environment variables will be used.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key. If it is None, the OPENAI_API_KEY from the environment variables will be used.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", args.base_url),
        api_key=os.environ.get("OPENAI_API_KEY", args.api_key),
    )
    if args.template.endswith(".txt") and os.path.exists(args.template):
        with open(args.template, "r") as f:
            args.template = "".join(f.readlines())
    # print(f"Beautiful prompt template: {args.template}")

    for _ in range(args.max_retry_nums):
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.template + "\n" + str(args.prompt)}
            ],
            temperature=0.7,
            top_p=1,
            max_tokens=1024,
        )

        output = completion.choices[0].message.content
        output = extract_output(output, prefix='"detailed description": ')
        if output is not None:
            break
    print(f"Beautiful prompt: {output}")


if __name__ == "__main__":
    main()