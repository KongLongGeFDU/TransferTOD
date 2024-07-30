from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from openai import OpenAI
import torch
import json
import random
import time
import argparse
import re


def process_test_data(test_file):
    with open(test_file, encoding="utf-8") as f:
        raw_test_data = json.load(f)
    if isinstance(raw_test_data, dict):
        test_data = []
        for domain, conversations in raw_test_data["clean_gpt_human"].items():
            test_data += conversations
        for domain, conversations in raw_test_data["noise_gpt_human"].items():
            test_data += conversations
    else:
        test_data = raw_test_data
    return test_data


def load_lora_model(model_name_or_path, adapter_name_or_path):
    device="cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()

    model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    return model, tokenizer


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    if "chatglm" not in model_path:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    return model, tokenizer


def get_response(model, tokenizer, user_input, ckpt_path, icl, icl_type):
    domain = re.search(r"需要信息抽取的场景是“(.*?)”", user_input).group(1)
    extract_slot = re.search(r"每一次提问的槽位数量为(.)", user_input).group(1)
    # icl处理待定
    if "clean" in ckpt_path:
        messages = []
        if icl:
            for conv in get_icl_examples(icl_type):
                messages.append({"role": "user", "content": conv[0]})
                messages.append({"role": "assistant", "content": conv[1]})
        messages.append({"role": "user", "content": user_input})
        print("MESSAGES:")
        print(messages)
        response = model.chat(tokenizer, messages)
    elif "Baichuan2" in ckpt_path:
        messages = []
        if icl:
            for conv in get_icl_examples(icl_type):
                messages.append({"role": "user", "content": conv[0]})
                messages.append({"role": "assistant", "content": conv[1]})
            messages.append({"role": "user", "content": user_input})
        else:
            messages.append({"role": "system", "content": get_icl_examples(icl_type).replace("<domain>", domain).replace("<extract_slot>", extract_slot)})
            messages.append({"role": "user", "content": user_input[user_input.find("\n槽位")+1:]})
        print("MESSAGES:")
        print(messages)
        response = model.chat(tokenizer, messages)
    elif "Yi" in ckpt_path:
        messages = []
        if icl:
            for conv in get_icl_examples(icl_type):
                messages.append({"role": "user", "content": conv[0]})
                messages.append({"role": "assistant", "content": conv[1]})
            messages.append({"role": "user", "content": user_input})
        else:
            messages.append({"role": "system", "content": get_icl_examples(icl_type).replace("<domain>", domain).replace("<extract_slot>", extract_slot)})
            messages.append({"role": "user", "content": user_input[user_input.find("\n槽位")+1:]})
        print("MESSAGES:")
        print(messages)
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    elif "Qwen" in ckpt_path:
        if icl:
            history = []
            for conv in get_icl_examples(icl_type):
                history.append((conv[0], conv[1]))
            response, _ = model.chat(tokenizer, user_input, history=history)
            print("HISTORY:")
            print(history)
        else:
            sys_prompt = get_icl_examples(icl_type).replace("<domain>", domain).replace("<extract_slot>", extract_slot)
            print("SYSTEM:")
            print(sys_prompt)
            print("USER:")
            print(user_input[user_input.find("\n槽位")+1:])
            response, _ = model.chat(tokenizer, user_input[user_input.find("\n槽位")+1:], system=sys_prompt, history=None)
    elif "chatglm" in ckpt_path:
        messages = []
        if icl:
            for conv in get_icl_examples(icl_type):
                messages.append({"role": "user", "content": conv[0]})
                messages.append({"role": "assistant", "metadata": "", "content": conv[1]})
            print("HISTORY:")
            print(messages)
            response, _ = model.chat(tokenizer, user_input, history=messages)
        else:
            messages.append({"role": "system", "content": get_icl_examples(icl_type).replace("<domain>", domain).replace("<extract_slot>", extract_slot)})
            print("HISTORY:")
            print(messages)
            print("USER:")
            print(user_input[user_input.find("\n槽位")+1:])
            response, _ = model.chat(tokenizer, user_input[user_input.find("\n槽位")+1:], history=messages)
    elif "BlueLM" in ckpt_path:
        prompt = ""
        if icl:
            for conv in get_icl_examples(icl_type):
                prompt += f"[|Human|]:{conv[0]}[|AI|]:{conv[1]}</s>"
            prompt += f"[|Human|]:{user_input}[|AI|]:"
        else:
            p = get_icl_examples(icl_type).replace("<domain>", domain).replace("<extract_slot>", extract_slot) + "\n\n输入：\n" + user_input[user_input.find("\n槽位")+1:]
            prompt += f"[|Human|]:{p}[|AI|]:"
        print("PROMPT:")
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to("cuda")
        pred = model.generate(**inputs, max_new_tokens=1024)
        response = tokenizer.decode(pred.cpu()[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def switch_key(keys, api_key):
    if len(keys) == 0:
        return None
    elif len(keys) == 1:
        print("No other keys available, waiting for 5s")
        time.sleep(5)
        key = keys[0]
        return key
    else:
        key = random.choice(keys)
        while key == api_key:
            key = random.choice(keys)
        return key


def gpt_chat(user_input, raw_key_list, model, icl, icl_type):
    keys = raw_key_list
    random.seed(int(time.time()))
    api_key = random.choice(keys)
    messages = []
    if icl:
        for conv in get_icl_examples(icl_type):
            messages.append({"role": "user", "content": conv[0]})
            messages.append({"role": "assistant", "content": conv[1]})
        messages.append({"role": "user", "content": user_input})
    else:
        domain = re.search(r"需要信息抽取的场景是“(.*?)”", user_input).group(1)
        extract_slot = re.search(r"每一次提问的槽位数量为(.)", user_input).group(1)
        messages.append({"role": "system", "content": get_icl_examples(icl_type).replace("<domain>", domain).replace("<extract_slot>", extract_slot)})
        messages.append({"role": "user", "content": user_input[user_input.find("\n槽位")+1:]})
    print("MESSAGES:")
    print(messages)
    while True:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages
            ).choices[0].message.content
            return response
        except Exception as e:
            if "RateLimitError" in repr(e) or "APIConnectionError" in repr(e) or "AuthenticationError" in repr(e):
                if "per min" in repr(e):
                    pass
                elif "current quota" in repr(e):
                    keys.remove(api_key)
                api_key = switch_key(keys, api_key)
                if api_key is None:
                    print("All the keys are expired.")
                    exit(0)
            else:
                print("Unknown error.")
                print(repr(e))


def get_icl_examples(icl_type):
    with open("./examples.json", encoding="utf-8") as f:
        examples = json.load(f)
    if icl_type == "sys":
        icl_examples = examples[icl_type][0]["conversations"][0]["value"]
    else:
        icl_examples = []
        for conv in examples[icl_type]:
            icl_examples.append([conv["conversations"][0]["value"], conv["conversations"][1]["value"]])
    return icl_examples


def inference(base_path, lora, ckpt_path, test_file, results_dir, api_key, icl, icl_type):
    if lora:
        model, tokenizer = load_lora_model(base_path, ckpt_path)
    elif ckpt_path in ["gpt-3.5-turbo", "gpt-4-1106-preview"]:
        raw_key_list = []
        raw_key_list.append(api_key)
    else:
        model, tokenizer = load_model(ckpt_path)

    test_data = process_test_data(test_file)
    
    # 开始对话
    for idx, item in enumerate(test_data):
        user_input = item["conversations"][0]["value"]
        print("-------------------------------------")
        print(str(idx+1) + '/' + str(len(test_data)))
        print("User: " + user_input)
        
        if ckpt_path in ["gpt-3.5-turbo", "gpt-4-1106-preview"]:
            response = gpt_chat(user_input, raw_key_list, ckpt_path, icl, icl_type)
        else:
            response = get_response(model, tokenizer, user_input, ckpt_path, icl, icl_type)
        
        if isinstance(response, dict):
            print("response is a dict")
            response = str(response)
        print("MODEL: " + response)
        print("\n")
        conv = [{"from": "human", "value": user_input}, {"from": "gpt", "value": response}]

        with open(results_dir, "a", encoding="utf-8") as w:
            w.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, default="./Baichuan2-7B-Base")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument("-d", "--test_data", type=str)
    parser.add_argument("-r", "--result_file", type=str)
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--icl_type", type=str, default="sys", help="id/ood-28/ood-29/ood-30")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    inference(args.base, args.lora, args.checkpoint, args.test_data, args.result_file, args.api_key, args.icl, args.icl_type)
