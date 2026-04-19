# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
import json
import re
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import rlla_token_call_token

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

tool_variants = json.load(open("dataset/tools.json", "r"))
tool_doc_dict_token = {tool_doc["token"]: {"token": tool_doc["token"], "description": tool_doc["description"], "parameters": tool_doc["parameters"]} for tool_doc in tool_variants.values()}
tool_doc_dict_name = {tool_doc["token"].strip('<>'): {"name": tool_doc["token"].strip('<>'), "description": tool_doc["description"], "parameters": tool_doc["parameters"]} for tool_doc in tool_variants.values()}

class RllaInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of rlla.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())  # UUID（通用唯一识别码）在Python中用于生成唯一标识，包括uuid1 ()（基于时间戳和MAC地址），uuid3 ()（基于MD5命名空间和字符串），uuid4 ()（随机生成），uuid4()由于其随机性和高唯一性，最常被使用
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
                
        # self._instance_dict[instance_id]["response"] = content
        self._instance_dict[instance_id]["response"] += content
        should_terminate_sequence, response, reward = False, "", 0.0

        if "<tool_token>" in self._instance_dict[instance_id]["ground_truth"]:
            tool_type = "token"
            tool_doc_dict = tool_doc_dict_token
        elif "<tool_name>" in self._instance_dict[instance_id]["ground_truth"]:
            tool_type = "name"
            tool_doc_dict = tool_doc_dict_name
        else:
            tool_type = ""
            tool_doc_dict = {}
        tool_token_match = re.search(rf'<tool_{tool_type}>\n(.*?)\n</tool_{tool_type}>', content, re.DOTALL)
        tool_call_match = re.search(r'<tool_call>\n(.*?)\n</tool_call>', content, re.DOTALL)
        if tool_token_match:
            tool_doc_entry = []
            tool_tokens_pd = tool_token_match.group(1).strip()
            tool_tokens_pd_lf = [token.strip() for token in tool_tokens_pd.split("\n")]
            for tool_token_pd in tool_tokens_pd_lf:
                if tool_token_pd in tool_doc_dict.keys():
                    tool_doc_entry.append(tool_doc_dict[tool_token_pd])
            response = f"Now invoke the tool call. Here is the tool documentation: " + "\n".join([json.dumps(tool_doc) for tool_doc in tool_doc_entry])
            should_terminate_sequence = False
        elif tool_call_match:
            response = "Tool call finished!"
            should_terminate_sequence = True

        # reward = await self.calculate_score(instance_id)
        # if reward == 1.0:
        #     response = "Your response is correct!"
        #     should_terminate_sequence = True
        # else:
        #     response = "Your response is incorrect! You need to reflect on your answer and try again."
        #     should_terminate_sequence = False

        return should_terminate_sequence, response, reward, {}

    # async def calculate_score(self, instance_id: str, **kwargs) -> float:
    #     return gsm8k.compute_score(
    #         self._instance_dict[instance_id]["response"],
    #         self._instance_dict[instance_id]["ground_truth"],
    #         method="strict",
    #         format_score=0.0,
    #         score=1.0,
    #     )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
