# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import os
from collections import Counter


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if sorted(list1) == sorted(list2):
        return 1.0
    
    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if sorted(list1) != sorted(list2):
            return 0.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0
    

# custoimzed reward functions: format
def customize_format_reward_func(completions, answer, step, tool_type, **kwargs):
    max_possible_reward = 1.0
    min_possible_reward = 0.0
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0
    
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    
    # if step % 40 == 0:
    #     print("\n======= Answer ======= ")
    #     print(answer[0])
    #     print("======= Responses ======= ")
    #     for idx, response in enumerate(responses):
    #         print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        pattern_think = r"<think>(.*?)</think>"
        pattern_tool_token = rf"<tool_{tool_type}>.*?</tool_{tool_type}>" # (.*?) 和 .*? 在正则表达式中效果相同，都是非贪婪匹配任意字符（包括空字符），区别只在于括号：(.*?) 会捕获匹配内容，结果可用 group(1) 取出。.*? 不会捕获内容，无法通过分组获取。
        pattern_tool_call = r"<tool_call>.*?</tool_call>" # ^ 表示匹配字符串的开头，$ 表示匹配字符串的结尾。加上它们，正则表达式只会匹配整个字符串完全符合 <tool_call>...</tool_call> 的情况。去掉后，可以匹配字符串中任意位置出现的 <tool_call>...</tool_call> 片段。

        reward = min_possible_reward
        if re.search(pattern_tool_token, ans, re.DOTALL) and re.search(pattern_tool_call, ans, re.DOTALL):
            if re.search(pattern_think, response, re.DOTALL) and re.search(pattern_tool_token, response, re.DOTALL) and re.search(pattern_tool_call, response, re.DOTALL):
                reward = max_possible_reward
        elif re.search(pattern_tool_call, ans, re.DOTALL):
            if re.search(pattern_think, response, re.DOTALL) and re.search(pattern_tool_call, response, re.DOTALL):
                reward = max_possible_reward
        else:
            raise ValueError(f"Unknown tool type: {ans}")
        rewards.append(reward)
        
    # if step % 40 == 0:
    #     print("Reward for <format>: ", rewards)
    # print("\n======= Reward for <format> =======")
    # print("Reward function for <format> is called ...")
    # print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(completions, answer, step, tool_type, **kwargs):
    max_possible_reward = 1.0
    min_possible_reward = 0.0
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512
    
    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0
        
        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)
    
    if step % 40 == 0:
        print("Reward for <length>: ", rewards)
    # print("\n======= Reward for <length> =======")
    # print("Reward function for <length> is called ...")
    # print(rewards)
    return rewards
                

def compute_tool_call_reward(tool_calls_gt, tool_calls_pd, max_possible_reward, min_possible_reward, tool_type):
    if tool_calls_gt == tool_calls_pd:
        return max_possible_reward
    
    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if sorted(tool_calls_gt) != sorted(tool_calls_pd):
            return min_possible_reward

    score_tools = 0.0
    score_tool_calls = 0.0

    tools_gt = [tool_call[tool_type] for tool_call in tool_calls_gt]
    tools_pd = [tool_call[tool_type] for tool_call in tool_calls_pd]
    score_tools = match_score(list(tools_gt), list(tools_pd))
    
    local_max_possible = 0.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for tool_call_gt in tool_calls_gt:
        tool_call_tool_gt = tool_call_gt.get(tool_type, "")
        tool_call_params_gt = tool_call_gt.get("parameters", {})
        
        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0# + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, tool_call_pd in enumerate(tool_calls_pd):
            # if i in used_pd_indices or pd_tool["tool_token"] != gt_name:    # 这个条件是说，如果pd_tool已经被匹配过了，或者pd_tool的tool_token和gt_name不匹配，则跳过（放大了tool_token选对的重要性）
            if i in used_pd_indices or tool_call_pd.get(tool_type, "") != tool_call_tool_gt:
                continue
            
            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if sorted(tool_call_gt) == sorted(tool_call_pd):
                    best_match = tool_call_pd
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue
            
            # pd_name = pd_tool.get(tool_type, "")
            # if pd_name == gt_name:
            #     name_score = 1.0
            # else:
            #     name_score = 0.0
            
            tool_call_params_pd = tool_call_pd.get("parameters", {})

            # 将gt_params和pd_params转换为hashable type的list（如tuple），这样做是因为dict的items()返回的是dict_items对象，虽然可以转为list进行比较，但其中的value如果是不可哈希类型（如list、dict等），在Counter等操作中会报错。转为tuple后，每个参数对(key, value)都变成了可哈希类型，便于后续的集合/计数操作。
            tool_call_params_gt_list = [(k, json.dumps(v, sort_keys=True)) for k, v in tool_call_params_gt.items()]
            tool_call_params_pd_list = [(k, json.dumps(v, sort_keys=True)) for k, v in tool_call_params_pd.items()]
            score_params = match_score(tool_call_params_gt_list, tool_call_params_pd_list)
            # param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # # Calculate correctness score for parameter values
            # score_correctness = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            # total_score = score_tool_call + score_params# + score_correctness
            total_score = score_params
            # print("score_tool_call:", score_tool_call)
            # print("score_params:", score_params)
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = tool_call_pd
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score_tool_calls += best_match_score


    assert float(len(tool_calls_gt)) == local_max_possible
    return score_tools + score_tool_calls / local_max_possible
    # return (max_possible_reward - min_possible_reward) * (score_tools + score_tool_calls) / local_max_possible + min_possible_reward

'''
def compute_tool_token_param_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward, tool_type):
    if gt_tools == pd_tools:
        # print("Max possible score:", "Exact Match!")
        # print("Score:", max_possible_reward)
        return max_possible_reward
    
    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["tool_token"] for tool in gt_tools]
    pd_names = [tool["tool_token"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))
    
    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["tool_token"]
        gt_params = gt_tool["parameters"]
        
        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            # if i in used_pd_indices or pd_tool["tool_token"] != gt_name:    # 这个条件是说，如果pd_tool已经被匹配过了，或者pd_tool的tool_token和gt_name不匹配，则跳过（放大了tool_token选对的重要性）
            if i in used_pd_indices:
                continue
            
            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue
            
            pd_params = pd_tool["parameters"]
            sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_calls_gt]) == sorted([json.dumps(tc.get("parameters", {}), sort_keys=True) for tc in tool_calls_pd])
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    # print()
    # print("Max possible score:", local_max_possible)
    # print("Score:", score)
    
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward
'''

# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(completions, answer, step, tool_type, **kwargs):
    max_possible_reward_token = 1.0
    min_possible_reward_token = 0.0
    max_possible_reward_tool_call = 2.0
    min_possible_reward_tool_call = 0.0

    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        max_possible_reward = 1.0
        min_possible_reward = -1.0
    else:
        max_possible_reward = 2.0
        min_possible_reward = -2.0

    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0
    
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        reward = 0.0

        tool_token_match_gt = re.search(rf'<tool_{tool_type}>\n(.*?)\n</tool_{tool_type}>', ans, re.DOTALL)
        tool_token_match_pd = re.search(rf'<tool_{tool_type}>\n(.*?)\n</tool_{tool_type}>', response, re.DOTALL)
        tool_call_match_gt = re.search(r'<tool_call>\n(.*?)\n</tool_call>', ans, re.DOTALL)
        tool_call_match_pd = re.search(r'<tool_call>\n(.*?)\n</tool_call>', response, re.DOTALL)
        
        '''
        if tool_token_match_gt:
            tool_tokens_gt = tool_token_match_gt.group(1).strip()
            tool_tokens_gt = [token.strip() for token in tool_tokens_gt.split("\n")]
            try:
                assert tool_token_match_pd
                tool_tokens_pd = tool_token_match_pd.group(1).strip()
                tool_tokens_pd_lf = [token.strip() for token in tool_tokens_pd.split("\n")]

                reward += (max_possible_reward_token - min_possible_reward_token) * match_score(tool_tokens_gt, tool_tokens_pd_lf) + min_possible_reward_token
            except:
                reward += min_possible_reward_token
        '''
        # print("reward_token:", reward)
        if tool_call_match_gt:
            tool_calls_gt = tool_call_match_gt.group(1).strip()
            tool_calls_gt = [json.loads(tool_call.strip()) for tool_call in tool_calls_gt.split("\n")]            
            try:
                assert tool_call_match_pd
                tool_calls_str_pd = tool_call_match_pd.group(1).strip()
                tool_calls_pd = [json.loads(tool_call.strip()) for tool_call in tool_calls_str_pd.split("\n")]
                reward += compute_tool_call_reward(tool_calls_gt, tool_calls_pd, max_possible_reward_tool_call, min_possible_reward_tool_call, tool_type)
            except:
                reward += min_possible_reward_tool_call
        

        rewards.append(reward)
    
    # if step % 40 == 0:
    #     print("Reward for <tool call>: ", rewards)
    # print("\n======= Reward for <tool token param> =======")
    # print("Reward function for <tool token param> correctness is called ...")
    # print(rewards)
    return rewards


def compute_score(solution_str, ground_truth, step=0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    completions = [[{"role": "assistant", "content": solution_str}]]
    # Toolrl
    # exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    # if "llama" in exp_name or "Llama" in exp_name:
    #     predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    # elif "qwen" in exp_name or "Qwen" in exp_name:
    #     predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    # else:
    #     raise NotImplementedError(f"Unknown model name: {exp_name}")
    
    # completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    if "<tool_token>" in ground_truth:
        tool_type = "token"
    elif "<tool_name>" in ground_truth:
        tool_type = "name"
    else:
        tool_type = ""
    
    format_score = customize_format_reward_func(completions, answer, step, tool_type)[0]
    correctness_score = customize_correctness_reward_tool(completions, answer, step, tool_type)[0]
    
    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, answer, step, tool_type)[0]
    else:
        length_score = 0
    
    
    score = format_score + correctness_score + length_score
    
    # if step % 3 == 0:
    #     print("solution_str:", solution_str)
    #     print("ground_truth:", ground_truth)
    #     print("fomrat_score:", fomrat_score)
    #     print("correctness_score:", correctness_score)
    #     # print("length_score:", length_score)
    #     print("score:", score)
    
    return score#, fomrat_score, correctness_score, length_score


if __name__ == "__main__":
    solution_str = "```json\n{\n  \"tool_token\": \"get_current_weather\",\n  \"parameters\": {\n    \"location\": \"San Francisco\"\n  }\n}\n```"
    ground_truth = "```json\n{\n  \"tool_token\": \"get_current_weather\",\n  \"parameters\": {\n    \"location\": \"San Francisco\"\n  }\n}\n```"
    step = 0


    solution_str = """
<tool_token>
<<education.charter_schedule>>
</tool_token>
user
Now invoke the tool call. Here is the tool documentation: {"token": "<<education.charter_schedule>>", "description": "Fetches the schedule for various classes and events in a charter school system.", "parameters": {"school_id": {"description": "Unique identifier for the charter school.", "type": "string", "default": ""}, "date": {"description": "Specific date to retrieve the schedule for.", "type": "string", "default": ""}, "activities": {"description": "List of activities to filter the schedule by.", "type": "array", "default": ""}}}
assistant
<tool_call>
{"token": "<<education.charter_schedule>>", "parameters": {"school_name": "Hilltop Academy", "date": "2023-01-01", "teachers": ["John Doe", "Jane Smith", "Michael Brown"]}}
{"token": "<<education.charter_schedule>>", "parameters": {"school_name": "Hilltop Academy", "date": "2023-01-02", "teachers": ["John Doe", "Jane Smith", "Michael Brown"]}}
{"token": "<<education.charter_schedule>>", "parameters": {"school_name": "Hilltop Academy", "date": "2023-01-03", "teachers": ["John Doe", "Jane Smith", "Michael Brown"]}}
</tool_call>
"""
    ground_truth = """
<tool_token>
<<education.charter_schedule>>
</tool_token><tool_call>
{"token": "<<education.charter_schedule>>", "parameters": {"school_id": "Hilltop Academy", "date": "2023-01-01", "activities": [{"activity_type": "Class", "details": {"instructor": "John Doe"}}, {"activity_type": "Class", "details": {"instructor": "Jane Smith"}}, {"activity_type": "Class", "details": {"instructor": "Michael Brown"}}]}}
{"token": "<<education.charter_schedule>>", "parameters": {"school_id": "Hilltop Academy", "date": "2023-01-02", "activities": [{"activity_type": "Class", "details": {"instructor": "John Doe"}}, {"activity_type": "Class", "details": {"instructor": "Jane Smith"}}, {"activity_type": "Class", "details": {"instructor": "Michael Brown"}}]}}
{"token": "<<education.charter_schedule>>", "parameters": {"school_id": "Hilltop Academy", "date": "2023-01-03", "activities": [{"activity_type": "Class", "details": {"instructor": "John Doe"}}, {"activity_type": "Class", "details": {"instructor": "Jane Smith"}}, {"activity_type": "Class", "details": {"instructor": "Michael Brown"}}]}}
</tool_call>
"""
    # score = compute_score(solution_str, ground_truth, step)
    # print("score:", score)

    solution_str = """
<tool_token>
<<album_tracks>>
<<Country Code Extractor>>
</tool_token>
user
Now invoke the tool call. Here is the tool documentation: {"token": "<<album_tracks>>", "description": "Fetches the tracks of a specified album from the Spotify API using RapidAPI.", "parameters": {"is_id": {"description": "The unique identifier for the album.", "type": "str", "default": "3IBcauSj5M2A6lTeffJzdv"}, "offset": {"description": "The starting point for the track list. Defaults to 0.", "type": "int, optional", "default": "0"}, "limit": {"description": "The maximum number of tracks to return. Defaults to 300.", "type": "int, optional", "default": "300"}}}
{"token": "<<Country Code Extractor>>", "description": "Extracts the country code and national number from an international phone number.", "parameters": {"phone": {"description": "The international phone number in the format `+XX XXXXXXXXXX`.", "type": "string", "default": ""}}}
assistant
<tool_call>
{"token": "<<album_tracks>>", "parameters": {"is_id": "67890", "limit": 50}}
{"token": "<<Country Code Extractor>>", "parameters": {"phone": "+441234567890"}}
</tool_call>
"""
    ground_truth = """
<tool_token>
<<album_tracks>>
<<analysis>>
</tool_token><tool_call>
{"token": "<<album_tracks>>", "parameters": {"is_id": "67890", "limit": 50}}
{"token": "<<analysis>>", "parameters": {"telephone": "+441234567890", "country": "UK"}}
</tool_call>
    """
    score = compute_score(solution_str, ground_truth, step)
    print("score:", score)