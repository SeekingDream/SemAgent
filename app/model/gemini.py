"""
For models other than those from OpenAI, use LiteLLM if possible.
"""

import json
import os
import sys
from typing import Literal

import litellm
from litellm.utils import Choices, Message, ModelResponse
from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential
import vertexai

from app.log import log_and_print
from app.model import common
from app.model.common import Model

from google import genai
from google.genai.types import HttpOptions
import warnings

from google.genai import types
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
from vertexai.generative_models import (
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Part,
    SafetySetting,
)


class GeminiModel(Model):
    """
    Base class for creating Singleton instances of Gemini models.
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
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        key_name = "GEMINI_API_KEY"
        credential_name = "GOOGLE_APPLICATION_CREDENTIALS"

        gemini_key = os.getenv(key_name)
        credential_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not (gemini_key or credential_key):
            print(f"Please set the {key_name} or {credential_name} env var")
            sys.exit(1)
        return gemini_key or credential_key

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        **kwargs,
    ):
        # FIXME: ignore tools field since we don't use tools now
        try:
            prefill_content = "{"
            if response_format == "json_object":  # prefill
                messages.append({"role": "assistant", "content": prefill_content})

            response = litellm.completion(
                model=self.name,
                messages=messages,
                temperature=common.MODEL_TEMP,
                max_tokens=1024,
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
            if response_format == "json_object":
                # prepend the prefilled character
                if not content.startswith(prefill_content):
                    content = prefill_content + content

            return content, cost, input_tokens, output_tokens

        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e


class GeminiPro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini-1.0-pro-002", 0.00000035, 0.00000105, parallel_tool_call=True
        )
        self.note = "Gemini 1.0 from Google"


class Gemini15Pro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini-1.5-pro-preview-0409",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 from Google"



class GeminiModelGeneric(Model):
    """
    Base class for creating Singleton instances of Gemini models,
    That does not use LiteLLM and Google API key.
    ADC needs to be set up with the correct project then this will work.
    follow the steps in:
    1) https://cloud.google.com/sdk/docs/authorizing
    2) https://cloud.google.com/docs/authentication/provide-credentials-adc
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
        parallel_tool_call: bool = False,
        locations: list[str] = ['us-central1']
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self._initialized = True
        self.name = name
        self.client = GenerativeModel(self.name)
        self.seed = None
        self.locations = locations
        self.vertexai_init_location('us-central1')

        warnings.filterwarnings("ignore", 
                        "Your application has authenticated using end user credentials",
                        UserWarning)

    
    def setup(self) -> None:
        """
        Check if setup is dine.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        project_name = "GOOGLE_CLOUD_PROJECT"
        location_name = "GOOGLE_CLOUD_LOCATION"
        use_vertexai_name = "GOOGLE_GENAI_USE_VERTEXAI"

        project = os.getenv(project_name)
        location = os.getenv(location_name)
        use_vertexai = os.getenv(use_vertexai_name)

        if not project or not location or not use_vertexai:
            print(f"""Please set the {project_name} or {location_name} or {use_vertexai_name} env vars.""")
            sys.exit(1)

    def generate_safety_config(self):
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            )
        ]
    
    def get_system_instructions_and_contents_from_messages(self, messages: list[dict]):
        system_instructions = "You are a helpful assistant."
        contents: list[Content] = []

        for message in messages:
            role, content =  message["role"], message["content"]
            if role == "system":
                system_instructions = content
            elif role == "user":
                contents.append(
                    Content(role="user", parts=[Part.from_text(content)])
                    )
            elif role == "assistant":
                contents.append(
                    Content(role="model", parts=[Part.from_text(content)])
                    )
            else:
                contents.append(
                    Content(role="user", parts=[Part.from_text(content)])
                    )
        return system_instructions, contents
    
    def extract_responses(self, candidate_list) :
        """ Returns a list of responses from the LLM. """
        response_list = []
        for index in range(len(candidate_list)) :
            candidate = candidate_list[index]
            try :
                role = candidate.content.role
                response = candidate.content.parts[-1].text
                response_list.append(response)
            except Exception as e:
                pass
        return response_list
    
    def vertexai_init_location(self, location:str):
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=location)

    @retry(wait=wait_random_exponential(min=60*0.1, max=60*30), stop=stop_after_attempt(10))
    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        num_candidates=1,
        response_mime_type=None,
        temperature=common.MODEL_TEMP,
        **kwargs,
    ):
        
        #prefill_content = "IMPORTANT: Your response must be a **complete, valid JSON object** starting with '{' and ending with '}'. that can be parsed with `json.loads(output)"

        return_only_one_output = True if num_candidates==1 else False
        
        if response_format == "json_object" and return_only_one_output: 
            #messages.append({"role": "assistant", "content": prefill_content})
            num_candidates=8
            temperature=0.4
            response_mime_type = "application/json"

        generation_config = GenerationConfig(
            temperature=temperature, 
            top_p=top_p, 
            candidate_count=num_candidates if num_candidates<=8 else 8, 
            seed=self.seed,
            response_mime_type=response_mime_type,
            )

        # FIXME: ignore tools field since we don't use tools now
        for location in self.locations:
            try:
                self.vertexai_init_location(location)

                system_instructions, contents = self.get_system_instructions_and_contents_from_messages(messages)

                self.client._system_instruction = system_instructions
                response = self.client.generate_content(
                            contents = contents, 
                            generation_config = generation_config,
                            safety_settings = self.generate_safety_config()
                            )

                #cost calculations
                resp_usage = response.usage_metadata
                assert resp_usage is not None
                input_tokens = int(resp_usage.prompt_token_count)
                output_tokens = int(resp_usage.candidates_token_count)
                cost = self.calc_cost(input_tokens, output_tokens)

                common.thread_cost.process_cost += cost
                common.thread_cost.process_input_tokens += input_tokens
                common.thread_cost.process_output_tokens += output_tokens

                #extracting the response
                all_content = self.extract_responses(response.candidates)

                if response_format == "json_object":
                    for content in all_content:
                        if return_only_one_output:
                            try:
                                test_if_it_loads_into_a_json = json.loads(content)
                                return str(content), cost, input_tokens, output_tokens
                            except Exception as e:
                                continue
                    return str(all_content[-1]), cost, input_tokens, output_tokens

                if len(all_content)==0:
                    content = "None"
                elif len(all_content)==1:
                    content =  all_content[0]
                else:
                    if return_only_one_output:
                        content = str(all_content[-1])
                    else:
                        content = all_content

                if content is None:
                    content = "None" # Just to make sure addition to a string is valid

                return content, cost, input_tokens, output_tokens

            except Exception as e:
                log_and_print(f"Error: {e} at location: {location}")
                last_error = e
                continue
        raise last_error

class Gemini15ProGeneric(GeminiModelGeneric):
    def __init__(self):
        super().__init__(
            "gemini-1.5-pro",
            0.00000125,
            0.000005,
            parallel_tool_call=True,
            locations = ['us-central1', 'northamerica-northeast1', 'us-east5', 'us-south1', 'us-west1', 'us-east4', 'us-east1', 'us-west4', 'australia-southeast1', 'australia-southeast2', 'us-west3', 'northamerica-northeast2', 'us-west2']
        )
        self.note = "Gemini 1.5 from Google but this implementation does not use LiteLLM"

class Gemini25ProGeneric(GeminiModelGeneric):
    def __init__(self):
        super().__init__(
            "gemini-2.5-pro-preview-05-06",
            0.0000025,
            0.000015,
            parallel_tool_call=True,
            locations = ['us-central1']
        )
        # log_and_print("setting up the gemini-2.5-pro-preview-05-06 model")
        self.note = "Gemini 2.5 pro - most advanced reasoning gemini model - from Google but this implementation does not use LiteLLM"

class Gemini25ProGenericActual(GeminiModelGeneric):
    def __init__(self):
        super().__init__(
            "gemini-2.5-pro",
            0.0000025,
            0.000015,
            parallel_tool_call=True,
            locations = ['us-central1']
        )
        # log_and_print("setting up the gemini-2.5-pro model")
        self.note = "Gemini 2.5 pro - most advanced reasoning gemini model - from Google but this implementation does not use LiteLLM and is not deprecated"


