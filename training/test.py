import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from models.loading import load_model, load_tokenizer
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import re
import argparse
import os
import deepspeed
from collections import defaultdict
from fastchat.conversation import get_conv_template
import torch._dynamo

torch._dynamo.config.suppress_errors = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_tool_memorization(results, type=None):
    correct = 0
    for entry in results:
        if entry["predicted_text"].strip() == entry["target_text"].strip():
            correct += 1
    accuracy = correct / len(results)
    print(f"Tool memorization accuracy: ({correct}/{len(results)}) {accuracy:.2%}")

def evaluate_tool_recall(results, type=None):
    correct = 0
    for entry in results:
        if entry["predicted_text"].strip() == entry["target_text"].strip():
            correct += 1
    accuracy = correct / len(results)
    print(f"Tool recall accuracy: ({correct}/{len(results)}) {accuracy:.2%}")

def evaluate_tool_tokens(results, type=None):
    scores_tool_tokens_lf = 0
    scores_tool_tokens_format = 0
    scores_tool_tokens_subset = 0
    scores_responses = 0
    total_tool_tokens = 0
    total_tool_tokens_r1_qwen = 0
    total_responses = 0

    for entry in results:
        tool_token_match = re.search(r'<tool_token>\n(.*?)\n</tool_token>', entry["target_text"], re.DOTALL)
        if tool_token_match:
            tool_tokens_gt = tool_token_match.group(1).strip()
            tool_tokens_gt = tool_tokens_gt.strip().split("\n")
        else:
            tool_tokens_gt = []

        tool_token_match = re.search(r'<tool_token>\n(.*?)\n</tool_token>', entry["predicted_text"], re.DOTALL)
        tool_token_match_r1_qwen = re.search(r'</think>\n(.*)', entry["predicted_text"], re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', entry["predicted_text"], re.DOTALL)

        if len(tool_tokens_gt) > 0:
            total_tool_tokens += 1
            if (tool_token_match or tool_token_match_r1_qwen) and not response_match:
                tool_tokens_pd = tool_token_match.group(1).strip() if tool_token_match else tool_token_match_r1_qwen.group(1).strip()
                if not tool_token_match and tool_token_match_r1_qwen:
                    total_tool_tokens_r1_qwen += 1
                tool_tokens_pd_lf = [token.strip() for token in tool_tokens_pd.split("\n")]
                tool_tokens_pd_comma1 = [token.strip() for token in tool_tokens_pd.split(",")]
                tool_tokens_pd_comma2 = [token.strip() for token in tool_tokens_pd.split(", ")]

                if sorted(tool_tokens_gt) == sorted(tool_tokens_pd_lf):
                    scores_tool_tokens_lf += 1
                if all(token in tool_tokens_pd_lf for token in tool_tokens_gt) or all(token in tool_tokens_pd_comma1 for token in tool_tokens_gt) or all(token in tool_tokens_pd_comma2 for token in tool_tokens_gt):
                    scores_tool_tokens_format += 1
                elif all(token in tool_tokens_pd for token in tool_tokens_gt):
                    print(f"Tool tokens predicted: {tool_tokens_pd}")
                    print(f"Tool tokens ground truth: {tool_tokens_gt}")
                    print("-"*100)
                if all(token in tool_tokens_pd for token in tool_tokens_gt):
                    scores_tool_tokens_subset += 1
        else:
            total_responses += 1
            if response_match and not tool_token_match:
                scores_responses += 1

    if total_tool_tokens > 0:
        print(f"Tool tokens accuracy (lf/format/subset): ({total_tool_tokens})", f"{scores_tool_tokens_lf/total_tool_tokens:.2%}/{scores_tool_tokens_format/total_tool_tokens:.2%}/{scores_tool_tokens_subset/total_tool_tokens:.2%}")
        print(f"Total tool tokens (r1 qwen): {total_tool_tokens_r1_qwen}")
    if total_responses > 0:
        print(f"Responses accuracy: ({total_responses})", f"{scores_responses/total_responses:.2%}")

def evaluate_tool_calls(results, type=None):
    scores_tool_calls = 0
    scores_tools = 0
    scores_parameters = 0
    scores_responses = 0
    total_tool_calls = 0
    total_tool_calls_r1_qwen = 0
    total_responses = 0
    for entry in results:
        if not entry["predicted_text"]:
            print(f"No predicted text found in entry: {entry['target_text']}")
            continue

        tool_call_match_gt = re.search(r'<tool_call>\n(.*?)\n</tool_call>', entry["target_text"], re.DOTALL)
        if tool_call_match_gt:
            tool_calls_gt = tool_call_match_gt.group(1).strip()
            tool_calls_gt = [json.loads(tool_call.strip()) for tool_call in tool_calls_gt.split("\n")]
        else:
            tool_calls_gt = []

        tool_call_match = re.search(r'<tool_call>\n(.*?)\n</tool_call>', entry["predicted_text"], re.DOTALL)
        tool_call_match_r1_qwen = re.search(r'</think>\n(.*)', entry["predicted_text"], re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', entry["predicted_text"], re.DOTALL)

        if len(tool_calls_gt) > 0:
            total_tool_calls += 1
            if (tool_call_match or tool_call_match_r1_qwen) and not response_match:
                tool_calls_str_pd = tool_call_match.group(1).strip() if tool_call_match else tool_call_match_r1_qwen.group(1).strip()
                if not tool_call_match and tool_call_match_r1_qwen:
                    total_tool_calls_r1_qwen += 1
                try:
                    tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                    scores_tool_calls += 1 if sorted([json.dumps(tc, sort_keys=True) for tc in tool_calls_gt]) == sorted([json.dumps(tc, sort_keys=True) for tc in tool_calls_pd]) else 0
                    scores_tools += 1 if sorted([tc.get("name", "") for tc in tool_calls_gt]) == sorted([tc.get("name", "") for tc in tool_calls_pd]) else 0
                    scores_parameters += 1 if sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_calls_gt]) == sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_calls_pd]) else 0
                except json.JSONDecodeError:
                    continue
        else:
            total_responses += 1
            if response_match and not tool_call_match:
                scores_responses += 1

    if total_tool_calls > 0:
        print(f"Tool calls/Tools/Parameters accuracy: ({total_tool_calls})", f"{scores_tool_calls/total_tool_calls:.2%}/{scores_tools/total_tool_calls:.2%}/{scores_parameters/total_tool_calls:.2%}")
        print(f"Total tool calls (r1 qwen): {total_tool_calls_r1_qwen}")
    if total_responses > 0:
        print(f"Total responses: {total_responses}")
        print(f"Responses accuracy:", f"{scores_responses/total_responses:.2%}")

def evaluate_tool_tokens_params(results, type=None):
    scores_tool_token_params = 0
    scores_tools = 0
    scores_parameters = 0
    scores_responses = 0
    total_tool_token_params = 0
    total_responses = 0
    for entry in results:
        if not entry["predicted_text"]:
            print(f"No predicted text found in entry: {entry['target_text']}")
            continue

        tool_token_param_match_gt = re.search(r'<tool_token_param>\n(.*?)\n</tool_token_param>', entry["target_text"], re.DOTALL)
        if tool_token_param_match_gt:
            tool_token_params_gt = tool_token_param_match_gt.group(1).strip()
            tool_token_params_gt = [json.loads(tool_token_param.strip()) for tool_token_param in tool_token_params_gt.split("\n")]
            entry["tool_token_params_gt"] = tool_token_params_gt
        else:
            entry["tool_token_params_gt"] = []

        tool_token_param_match = re.search(r'<tool_token_param>\n(.*?)\n</tool_token_param>', entry["predicted_text"], re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', entry["predicted_text"], re.DOTALL)

        if len(entry["tool_token_params_gt"]) > 0:
            total_tool_token_params += 1
            if tool_token_param_match and not response_match:
                tool_token_params_str_pd = tool_token_param_match.group(1).strip()
                try:
                    tool_token_params_pd = [json.loads(tool_token_param.strip()) for tool_token_param in tool_token_params_str_pd.split("\n")]
                    correct_tools = sorted([tc.get("tool_token", "") for tc in entry["tool_token_params_gt"]]) == sorted([tc.get("tool_token", "") for tc in tool_token_params_pd])
                    correct_parameters = sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in entry["tool_token_params_gt"]]) == sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_token_params_pd])
                    scores_tools += 1 if correct_tools else 0
                    scores_parameters += 1 if correct_parameters else 0
                    scores_tool_token_params += 1 if correct_tools and correct_parameters else 0
                except json.JSONDecodeError:
                    continue
        else:
            total_responses += 1
            if response_match and not tool_token_param_match:
                scores_responses += 1

    if total_tool_token_params > 0:
        print(f"Tool token params/Tools/Parameters accuracy: ({total_tool_token_params})", f"{scores_tool_token_params/total_tool_token_params:.2%}/{scores_tools/total_tool_token_params:.2%}/{scores_parameters/total_tool_token_params:.2%}")
    if total_responses > 0:
        print(f"Responses accuracy: ({total_responses})", f"{scores_responses/total_responses:.2%}")

def evaluate_tools_calls(results, type=None):
    tool_type = "name" if "semantic" in type or "w_tools" in type else "token"
    scores_tools_em, scores_f1, scores_tool_calls, scores_tools, scores_parameters, scores_responses = 0, 0, 0, 0, 0, 0
    total_tools_em, total_tool_calls, total_tool_calls_pd, total_tool_calls_pd_r1_qwen, total_tool_calls_pd_hammer, total_tool_calls_pd_xlam, total_tool_calls_pd_all_text, total_responses = 0, 0, 0, 0, 0, 0, 0, 0

    scores_tools_em_setting, scores_f1_setting, scores_tool_calls_setting, scores_tools_setting, scores_parameters_setting, scores_responses_setting = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    total_tools_em_setting, total_tool_calls_setting, total_responses_setting = defaultdict(int), defaultdict(int), defaultdict(int)
    
    for entry in results:
        if not entry["predicted_text"]:
            print(f"No predicted text found in entry: {entry['target_text']}")
            continue

        tool_token_match_gt = re.search(rf'<tool_{tool_type}>\n(.*?)\n</tool_{tool_type}>', entry["target_text"], re.DOTALL)
        tool_token_match_pd = re.search(rf'<tool_{tool_type}>\n(.*?)\n</tool_{tool_type}>', entry["predicted_text"], re.DOTALL)
        tool_call_match_gt = re.search(r'<tool_call>(.*?)</tool_call>', entry["target_text"], re.DOTALL)
        tool_call_match_pd = re.search(r'<tool_call>(.*?)</tool_call>', entry["predicted_text"], re.DOTALL)
        tool_call_match_pd_r1_qwen = re.search(r'</think>(.*)', entry["predicted_text"], re.DOTALL)
        tool_call_match_pd_hammer = re.search(r'```(.*?)```', entry["predicted_text"], re.DOTALL)
        # tool_call_match_pd_xlam = re.search(r'^(.*)$', entry["predicted_text"], re.DOTALL)
        response_match_gt = re.search(r'<response>(.*?)</response>', entry["target_text"], re.DOTALL)
        response_match_pd = re.search(r'<response>(.*?)</response>', entry["predicted_text"], re.DOTALL)
        
        if tool_token_match_gt:
            tool_tokens_gt = tool_token_match_gt.group(1).strip()
            tool_tokens_gt = [token.strip() for token in tool_tokens_gt.split("\n")]
            total_tools_em += 1
            if entry.get("setting", None) is not None:
                total_tools_em_setting[entry["setting"]] += 1
            if tool_token_match_pd and not response_match_pd:
                tool_tokens_pd = tool_token_match_pd.group(1).strip()
                tool_tokens_pd_lf = [token.strip() for token in tool_tokens_pd.split("\n")]
                precision = len([token for token in tool_tokens_pd_lf if token in tool_tokens_gt]) / len(tool_tokens_pd_lf) if len(tool_tokens_pd_lf) > 0 else 0
                recall = len([token for token in tool_tokens_gt if token in tool_tokens_pd_lf]) / len(tool_tokens_gt) if len(tool_tokens_gt) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                scores_f1 += f1
                if entry.get("setting", None) is not None:
                    scores_f1_setting[entry["setting"]] += f1

                if sorted(tool_tokens_pd_lf) == sorted(tool_tokens_gt):
                    scores_tools_em += 1
                    if entry.get("setting", None) is not None:
                        scores_tools_em_setting[entry["setting"]] += 1
                else:
                    entry["tools_em_f1"] = f"{0:.2%}/{f1:.2%}"
        
        if tool_call_match_gt:
            tool_calls_gt = tool_call_match_gt.group(1).strip()
            tool_calls_gt = [json.loads(tool_call.strip()) for tool_call in tool_calls_gt.split("\n")]            
            total_tool_calls += 1
            if entry.get("setting", None) is not None:
                total_tool_calls_setting[entry["setting"]] += 1
            
            if tool_call_match_pd:
                tool_calls_str_pd = tool_call_match_pd.group(1).strip()
                try:
                    tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                    total_tool_calls_pd += 1
                except:
                    try:
                        tool_calls_pd = json.loads('[' + tool_calls_str_pd + ']')   # '<tool_call>{"name": "<<live_giveaways_by_platform>>", "parameters": {"platform": "xbox-one"}}, {"name": "<<live_giveaways_by_platform>>", "parameters": {"platform": "android"}}</tool_call>'
                        total_tool_calls_pd_xlam += 1
                    except:
                        continue
            elif tool_call_match_pd_r1_qwen:
                tool_calls_str_pd = tool_call_match_pd_r1_qwen.group(1).strip()
                try:
                    tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                    total_tool_calls_pd_r1_qwen += 1
                except:
                    continue
            elif tool_call_match_pd_hammer:
                tool_calls_str_pd = tool_call_match_pd_hammer.group(1).strip()
                try:
                    tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                    total_tool_calls_pd_hammer += 1
                except:
                    continue
            else:
                tool_calls_str_pd = entry["predicted_text"].strip()
                try:
                    tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                    total_tool_calls_pd_all_text += 1
                except:
                    continue
            
            try:
                # tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                # Compare tool calls, tool names and parameters
                if sorted([json.dumps(tc, sort_keys=True) for tc in tool_calls_gt]) == sorted([json.dumps(tc, sort_keys=True) for tc in tool_calls_pd]):
                    scores_tool_calls += 1
                    if entry.get("setting", None) is not None:
                        scores_tool_calls_setting[entry["setting"]] += 1
                else: 
                    entry["tool_calls_em"] = 0
                if sorted([tc.get(tool_type, "") for tc in tool_calls_gt]) == sorted([tc.get(tool_type, "") for tc in tool_calls_pd]):
                    scores_tools += 1
                    if entry.get("setting", None) is not None:
                        scores_tools_setting[entry["setting"]] += 1
                if sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_calls_gt]) == sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_calls_pd]):
                    scores_parameters += 1
                    if entry.get("setting", None) is not None:
                        scores_parameters_setting[entry["setting"]] += 1
            except: # json.JSONDecodeError
                # print("Not valid JSON format")
                continue
            # else:
            #     # print(f"No tool_call tags found in output")    # for entry {entry['entry_id']}
            #     continue
        if response_match_gt and not tool_token_match_gt and not tool_call_match_gt:
            total_responses += 1
            if response_match_pd:
                scores_responses += 1

    if total_tools_em > 0:
        print(f"Tools EM accuracy/F1 score: ({total_tools_em})", f"{scores_tools_em/total_tools_em:.2%}/{scores_f1/total_tools_em:.2%}")
        for setting, total in total_tools_em_setting.items():
            print(f"Tools EM accuracy/F1 score: ({total})", f"{scores_tools_em_setting[setting]/total_tools_em_setting[setting]:.2%}/{scores_f1_setting[setting]/total_tools_em_setting[setting]:.2%}")

    if total_tool_calls > 0:
        print(f"Tool calls/Tools/Parameters accuracy: ({total_tool_calls})", f"{scores_tool_calls/total_tool_calls:.2%}/{scores_tools/total_tool_calls:.2%}/{scores_parameters/total_tool_calls:.2%}")
        for setting, total in total_tool_calls_setting.items():
            print(f"Tool calls/Tools/Parameters accuracy: ({total})", f"{scores_tool_calls_setting[setting]/total_tool_calls_setting[setting]:.2%}/{scores_tools_setting[setting]/total_tool_calls_setting[setting]:.2%}/{scores_parameters_setting[setting]/total_tool_calls_setting[setting]:.2%}")
        print(f"Total tool calls (pd): ({total_tool_calls_pd})")
        print(f"Total tool calls (r1 qwen): ({total_tool_calls_pd_r1_qwen})")
        print(f"Total tool calls (hammer): ({total_tool_calls_pd_hammer})")
        print(f"Total tool calls (xlam): ({total_tool_calls_pd_xlam})")
        print(f"Total tool calls (all text): ({total_tool_calls_pd_all_text})")

    if total_responses > 0:
        print(f"Total responses: {total_responses}")
        print(f"Responses accuracy:", f"{scores_responses/total_responses:.2%}")

def load_model_and_tokenizer(model_path, checkpoint_dir=None, virtual_tokens_file=None):
    if virtual_tokens_file is not None:
        # Load model and tokenizer
        tokenizer = load_tokenizer(
            model_path, 
            # cache_dir=args.cache_dir,
            virtual_tokens=True, 
            virtual_tokens_file=virtual_tokens_file,
        )

        model = load_model(
            model_path,
            architecture="causal",
            tokenizer=tokenizer,
            flash_attention=True,
            # cache_dir=args.cache_dir,
            virtual_tokens=True,
            virtual_tokens_file=virtual_tokens_file,
        )
    else:
        # if model_path == "MadeAgents/Hammer2.1-7b":
        #     cache_dir = "/data/models/"
        # elif model_path == "Salesforce/xLAM-7b-r":
        #     cache_dir = "/data/models/"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if checkpoint_dir is not None:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir) # already on cpu
        # model = model.cpu() # move to cpu
        model.load_state_dict(state_dict)

    model.eval()
    # model.to(device)  
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    print("world_size: ", world_size)

    if world_size > 1:
        ds_engine = deepspeed.init_inference(
            model=model,      # Transformers模型
            mp_size=world_size,        # GPU数量
            dtype=torch.float, # 权重类型(fp16) torch.float16
            replace_method="auto", # 让DS自动替换层
            replace_with_kernel_inject=True, # 使用kernel injector替换
        )
        model = ds_engine.module
    else:
        model.to(device)
        
    return model, tokenizer


def prepare_dataset(eval_file):
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    if "messages" in data[0].keys():
        format_test = lambda item: {
            "target_text": "".join([message["content"] for message in item["messages"] if message["role"] == "assistant"]),
            "messages": item["messages"][:next(i for i, m in enumerate(item["messages"]) if m.get("role") == "assistant")],
            "type": item["type"],
            "setting": item.get("setting", None)
        }
    else:
        format_test = lambda item: {
            "target_text": item["output"],
            "messages": [{"role": "system", "content": item["instruction"]}, {"role": "user", "content": item["input"]}],
            "type": item["type"],
            "setting": item.get("setting", None)
        }
    test_data = [format_test(item) for item in data]
    
    return test_data

def obtain_results_generation(test_data, model, tokenizer, batch_size=128, model_path=None):
    eos_token = "<|endoftext|>"
    
    # Evaluate each conversation in batches
    batch_size = min(batch_size, len(test_data))
    results = []
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        input_texts = []
        target_texts = []
        
        for item in batch:
            messages = item["messages"]
                # messages = [
                #     {"role": "system", "content": item["system_content"]},
                #     {"role": "user", "content": item["user_content"]}
                # ]
            if "xLAM-7b-r" in model_path or "mistral" in model_path:
                messages = [
                    {"role": "user", "content": messages[0]['content'] + "\n" + messages[1]['content']}
                ]   
            
            if "LLaMA-2" in model_path:
                conv = get_conv_template("llama-2")
                conv.set_system_message(messages[0]['content'])
                conv.append_message(conv.roles[0], messages[1]['content'])
                # conv.append_message(conv.roles[1], None)
                prompts = conv.get_prompt()
            else:
                prompts = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            input_texts.append(prompts)
        
        # Generate predictions
        # inputs = tokenizer(input_texts, padding=True, padding_side="left", return_tensors="pt").to(device)
        inputs = tokenizer(input_texts, padding=True, padding_side="left", return_tensors="pt").to(device)
        truncated_text = tokenizer.batch_decode(inputs.data["input_ids"], skip_special_tokens=True)

        terminators = [
            tokenizer.eos_token_id,
            # tokenizer.convert_tokens_to_ids(eos_token)
        ]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,    # 2048
                do_sample=False,
                eos_token_id=terminators,
            )
        
        predicted_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted_texts = [pred[len(trunc):] for pred, trunc in zip(predicted_texts, truncated_text)]
        # # if i == 0:
        # for id in range(len(predicted_texts)):
        #     print("Target output: ", target_texts[id])
        #     print("Predicted output: ", predicted_texts[id])
        #     print("-"*100)
        # break   #
        
        for item, predicted_text in zip(batch, predicted_texts):
            item["predicted_text"] = item.get("predicted_text", "") + predicted_text
            item["messages"].append({"role": "assistant", "content": predicted_text})
            results.append(item)

    return results

def process_results(results, tool_variants):
    tool_doc_dict = {tool_doc["token"]: {"token": tool_doc["token"], "description": tool_doc["description"], "parameters": tool_doc["parameters"]} for tool_doc in tool_variants.values()}
    for entry in results:
        tool_token_match = re.search(r'<tool_token>\n(.*?)\n</tool_token>', entry["predicted_text"], re.DOTALL)
        tool_doc_entry = []
        if tool_token_match:
            tool_tokens_pd = tool_token_match.group(1).strip()
            tool_tokens_pd_lf = [token.strip() for token in tool_tokens_pd.split("\n")]
            for tool_token_pd in tool_tokens_pd_lf:
                if tool_token_pd in tool_doc_dict.keys():
                    tool_doc_entry.append(tool_doc_dict[tool_token_pd])
        entry["messages"].append({"role": "user", "content": f"Now invoke the tool call. Here is the tool documentation: " + "\n".join([json.dumps(tool_doc) for tool_doc in tool_doc_entry])})

    return results

evaluate_functions = {
    "memorization": evaluate_tool_memorization,
    "recall": evaluate_tool_recall,
    "direct_call": evaluate_tool_calls,
    "think_call": evaluate_tool_calls,
    "call": evaluate_tool_calls,
    "direct_token": evaluate_tool_tokens,
    "think_token": evaluate_tool_tokens,
    "token": evaluate_tool_tokens,
    "token_w_tools": evaluate_tools_calls,
    "direct_token_param": evaluate_tool_tokens_params,
    "think_token_param": evaluate_tool_tokens_params,
    "token_param": evaluate_tool_tokens_params,
    # "token_param": evaluate_tool_calls,
    "token_param_w_tools": evaluate_tools_calls,
    "tool_call_w_tools": evaluate_tools_calls,
    "token_call_token": evaluate_tools_calls,
    "call_token": evaluate_tools_calls,
    "semantic_call_semantic": evaluate_tools_calls,
    "call_semantic": evaluate_tools_calls,

}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, required=False, help="The local rank.")
    parser.add_argument("--model_path", default=None, type=str, required=True, help="The model path.")
    parser.add_argument("--eval_file", default=None, type=str, required=True, help="The eval file.")
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=False, help="The checkpoint directory.")
    parser.add_argument("--virtual_tokens_file", default=None, type=str, required=False, help="The virtual tokens file.")
    parser.add_argument("--write_results", default=False, type=bool, required=False, help="Whether to write the results to a file.")
    parser.add_argument("--batch_size", default=128, type=int, required=False, help="The batch size.")
    parser.add_argument("--tool_file", default="dataset/tools.json", type=str, required=False, help="The tool file.")
    args = parser.parse_args()
    print(args)

    model_path = args.model_path
    eval_file = args.eval_file
    checkpoint_dir = args.checkpoint_dir
    virtual_tokens_file = args.virtual_tokens_file
    write_results = args.write_results
    batch_size = args.batch_size
    tool_file = args.tool_file

    tool_variants = json.load(open(tool_file, "r"))
    test_data = prepare_dataset(eval_file)

    model, tokenizer = load_model_and_tokenizer(model_path, checkpoint_dir, virtual_tokens_file)
    results = obtain_results_generation(test_data, model, tokenizer, batch_size=batch_size, model_path=model_path)
    if "token_call_token" in eval_file or "semantic_call_semantic" in eval_file:
        test_data = process_results(results, tool_variants)
        results = obtain_results_generation(test_data, model, tokenizer, batch_size=batch_size, model_path=model_path)

    if checkpoint_dir is not None:
        print("checkpoint_dir: ", checkpoint_dir)
    else:
        print("model_path: ", model_path)
    print("eval_file: ", eval_file)
    for type, evaluate_func in evaluate_functions.items():
        results_type = [result for result in results if result["type"] == type]
        if len(results_type) > 0:
            print(f"Evaluating {type}...")
            evaluate_func(results_type, type)

    results_file = None
    if write_results:
        model_tag = os.path.basename(model_path.rstrip("/"))
        eval_tag = os.path.splitext(os.path.basename(eval_file))[0]
        results_file = os.path.join("results", model_tag, f"{eval_tag}.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f)
