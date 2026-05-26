"""
For models other than those from OpenAI, use LiteLLM if possible.

Command:
PYTHONPATH=. python app/main.py swe-bench --model claude-3-haiku-20240307 --setup-map ../SWE-bench/setup_result/setup_map.json --tasks-map ../SWE-bench/setup_result/tasks_map.json --output-dir output --task django__django-11133 --model-temperature 0.3
PYTHONPATH=. python app/main.py swe-bench --model claude-3-haiku-20240307 --setup-map ../SWE-bench/setup_result/setup_map.json --tasks-map ../SWE-bench/setup_result/tasks_map.json --output-dir output --task-list-file /opt/SWE-bench/tasks.txt
"""

import os
import sys
from typing import Literal
import time

import litellm
from litellm.utils import Choices, Message, ModelResponse
from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.log import log_and_print
from app.model import common
from app.model.common import ClaudeContentPolicyViolation, Model
import tiktoken

import config
import json

class AnthropicModel(Model):
    """
    Base class for creating Singleton instances of Antropic models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        cost_per_input: float,
        cost_per_output: float,
        max_output_token: int = 4096,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self.max_output_token = max_output_token
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        key_name = "ANTHROPIC_API_KEY"
        key = os.getenv(key_name)
        if not key:
            print(f"Please set the {key_name} env var")
            sys.exit(1)
        return key

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ):
        # FIXME: ignore tools field since we don't use tools now
        if temperature is None:
            temperature = common.MODEL_TEMP

        try:

            if response_format == "json_object":
                last_content = messages[-1]["content"]
                last_content += "\nYour response should start with { and end with }. DO NOT write anything else other than the json."
                messages[-1]["content"] = last_content

            response = litellm.completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_output_token,
                top_p=top_p,
                stream=False,
            )

            assert isinstance(response, ModelResponse)
            resp_usage = response.usage
            assert resp_usage is not None
            input_tokens = int(resp_usage.prompt_tokens)
            output_tokens = int(resp_usage.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message
            content = self.extract_resp_content(resp_msg)

            return content, cost, input_tokens, output_tokens

        except litellm.exceptions.ContentPolicyViolationError:
            # claude sometimes send this error when writing patch
            log_and_print("Encountered claude content policy violation.")
            raise ClaudeContentPolicyViolation

        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e


class Claude3Opus(AnthropicModel):
    def __init__(self):
        super().__init__(
            "claude-3-opus-20240229", 0.000015, 0.000075, parallel_tool_call=True
        )
        self.note = "Most powerful model among Claude 3"


class Claude3Sonnet(AnthropicModel):
    def __init__(self):
        super().__init__(
            "claude-3-sonnet-20240229", 0.000003, 0.000015, parallel_tool_call=True
        )
        self.note = "Most balanced (intelligence and speed) model from Antropic"


class Claude3Haiku(AnthropicModel):
    def __init__(self):
        super().__init__(
            "claude-3-haiku-20240307", 0.00000025, 0.00000125, parallel_tool_call=True
        )
        self.note = "Fastest model from Antropic"


class Claude3_5Sonnet(AnthropicModel):
    def __init__(self):
        super().__init__(
            "claude-3-5-sonnet-20240620", 0.000003, 0.000015, parallel_tool_call=True
        )
        self.note = "Most intelligent model from Antropic"


class Claude3_5SonnetNew(AnthropicModel):
    def __init__(self):
        super().__init__(
            "claude-3-5-sonnet-20241022", 0.000003, 0.000015, parallel_tool_call=True
        )
        self.note = "Most intelligent model from Antropic"

#VertexAI code
#TODO: can move this to a new file specifically for VertexAI models

class VertexAIAnthropicModel(Model):
    """
    Base class for creating Singleton instances of Antropic models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        cost_per_input: float,
        cost_per_output: float,
        max_output_token: int = 4096,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self.max_output_token = max_output_token
        self._initialized = True #TODO: make this a config param
        self.sleep_for_a_fixed_duration_after_each_response = True
        self.sleep_if_overloaded = True
        self.time_to_sleep_after_each_response = 2
        self.vertex_locations = ["us-east5","europe-west1"]
        self.ENCODING = tiktoken.get_encoding("cl100k_base")

    def return_cred_json(self, cred_file_path):
        with open(cred_file_path, 'r') as file:
            vertex_credentials = json.load(file)

        vertex_credentials_json = json.dumps(vertex_credentials)
        return vertex_credentials_json

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()
        self.creds = self.return_cred_json(config.vertexai_creds_path)

    def check_api_key(self) -> str:
        assert config.vertex_project!="name of vertexai project", "please set the vertai project name accordingly"
        assert config.vertexai_creds_path!="<path to vertexai creds json>", "please set the vertai creds path accordingly"
        pass

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    def call(
        self,
        messages: list[dict],
        top_p=0.95,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        temperature=0.0,
        **kwargs,
    ):
        # FIXME: ignore tools field since we don't use tools now

        try:

            if response_format == "json_object":
                last_content = messages[-1]["content"]
                last_content += "\nYour response should start with { and end with }. DO NOT write anything else other than the json."
                messages[-1]["content"] = last_content

            if self.sleep_if_overloaded:
                attempt = 0
                response = None
                vertex_location_to_use = 0
                attempt_time_map = [0.1, 5, 5, 10, 30]
                max_tries = len(attempt_time_map)
                max_num_truncation_attempts = 3
                num_truncation_attempts = 0
                max_num_content_policy_violation = 3
                num_content_policy_violation = 0
                
                while attempt<max_tries and response is None:
                    try:
                        response = litellm.completion(
                                model = self.name,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=self.max_output_token,
                                top_p=top_p,
                                stream=False,
                                vertex_credentials=self.creds,
                                vertex_location=self.vertex_locations[vertex_location_to_use],
                                vertex_project=config.vertex_project
                            )
                        
                    except Exception as e:
                        error_to_string = str(e)
                        
                        # Overloaded/Resource Exhausted Error
                        if any(substring in error_to_string for substring in ["RESOURCE_EXHAUSTED", "Resource exhausted", "Overloaded", "overloaded", "UNAVAILABLE"]):
                            log_and_print(f"Overloaded error: Attempt {attempt} failed with model {self.vertex_locations[vertex_location_to_use]} - sleeping for {attempt_time_map[attempt]} minutes.")
                            if vertex_location_to_use == 0:
                                time.sleep(attempt_time_map[attempt]*60)
                                vertex_location_to_use = 1
                            else:
                                time.sleep(attempt_time_map[attempt]*60)
                                vertex_location_to_use = 0
                                attempt += 1

                        # Input too long TODO: make this more robust by summarization, not required for swebench lite
                        # As this temp fix should work for the call chain fixer
                        elif any(substring in error_to_string for substring in ["Prompt is too long", "exceeded"]):

                            # break out of try except if max truncation limits reached 
                            if num_truncation_attempts >= max_num_truncation_attempts:
                                attempt = max_tries
                            elif num_truncation_attempts < 4:
                                percent_left_after_truncation = 0.70
                            else:
                                percent_left_after_truncation = 0.50
                            log_and_print(f"Context Exceeded error: Input is too long, trying to reduce it by {percent_left_after_truncation}%.")
                            for i in range(len(messages)):
                                num_tokens_in_content_of_message = self.get_num_tokens_from_a_string(messages[i]["content"]) 
                                if num_tokens_in_content_of_message>50000:
                                    temp_num_chars = int(len(messages[i]["content"]) * percent_left_after_truncation / 2)
                                    messages[i]["content"] = messages[i]["content"][:temp_num_chars] + "..." + messages[i]["content"][-1*temp_num_chars:]
                            num_truncation_attempts += 1

                        elif any(substring in error_to_string for substring in ["ContentPolicyViolationError", "content filtering"]):
                            log_and_print("Encountered claude content policy violation.")
                            if num_content_policy_violation >= max_num_content_policy_violation:
                                attempt = max_tries                                
                            messages[-1]["content"] = messages[-1]["content"] + f". VERY IMPORTANT: Your previous {num_content_policy_violation} responses to this message were flagged by Anthropic as a content policy violation error. Please do not write anything that might trigger a content policy violation error."
                            if num_content_policy_violation%2==1:
                                if len(messages)>2:
                                    messages = messages[:-1]
                                else:
                                    messages[-1]["content"] = "skip this LLM call."
                            num_content_policy_violation += 1

                        else:
                            # if an un-encountered error occurs, then rety unti max_tries
                            time.sleep(attempt_time_map[attempt]*60)
                            attempt += 1
                            log_and_print(f"Unencountered LLM call failure, Error: {e}")

            else:
                response = litellm.completion(
                        model = self.name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_output_token,
                        top_p=top_p,
                        stream=False,
                        vertex_credentials=self.creds,
                        vertex_location=self.vertex_locations[0],
                        vertex_project=config.vertex_project
                    )

            if self.sleep_for_a_fixed_duration_after_each_response:
                time.sleep(self.time_to_sleep_after_each_response)

            assert isinstance(response, ModelResponse)

            resp_usage = response.usage
            assert resp_usage is not None

            input_tokens = int(resp_usage.prompt_tokens)

            output_tokens = int(resp_usage.completion_tokens)

            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message
            content = self.extract_resp_content(resp_msg)

            return content, cost, input_tokens, output_tokens

        except litellm.exceptions.ContentPolicyViolationError:
            # claude sometimes send this error when writing patch
            log_and_print("Encountered claude content policy violation.")
            raise ClaudeContentPolicyViolation

        except BadRequestError as e:
            print(e)
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e

    def convert_message_to_anthropic_input_message_format(self, messages: list[dict]) -> list:
        system = ""
        non_system_messages = []

        for i in messages:
            if i['role']=='system':
                system += i['content']
                system += " "
            else:
                non_system_messages.append(i)

        return [system, non_system_messages]
    
    def get_num_tokens_from_a_string(self, s: str) -> int:
        enc = self.ENCODING.encode(s)
        tokens = [self.ENCODING.decode([token]) for token in enc]
        return len(tokens)
    

# The models
class VertexAIClaude3_7SonnetNew(VertexAIAnthropicModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/claude-3-7-sonnet@20250219", 3/10**6, 15/10**6, parallel_tool_call=True
        )
        self.note = "The new Most intelligent Claude Sonnet 3.7 model from Antropic on VertexAI"

class VertexAIClaude3_5Sonnet(VertexAIAnthropicModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/claude-3-5-sonnet@20240620", 3/10**6, 15/10**6, parallel_tool_call=True
        )
        self.note = "Claude Sonnet 3.5 model from Antropic on VertexAI"

class VertexAIClaude3_5Haiku(VertexAIAnthropicModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/claude-3-5-haiku@20241022", 0.8/10**6, 4/10**6, parallel_tool_call=True
        )
        self.note = "Claude Haiku 3.5 model from Antropic on VertexAI"

class VertexAIClaude3Opus(VertexAIAnthropicModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/claude-3-opus@20240229", 15/10**6, 75/10**6, parallel_tool_call=True
        )
        self.note = "Claude Opus 3 model from Antropic on VertexAI"

class VertexAIClaude4Sonnet(VertexAIAnthropicModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/claude-sonnet-4@20250514", 3/10**6, 15/10**6, parallel_tool_call=True
        )
        self.note = "Claude Sonnet 4 model from Antropic on VertexAI"



