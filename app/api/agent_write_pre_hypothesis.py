"""
An agent, which is only responsible for the write_hypothesis tool call.
"""

import os
import json
import shutil
from collections.abc import Callable, Iterable
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app import globals
from app.api import agent_common
from app.api import validation
from app.data_structures import MessageThread, MethodId
from app.log import print_acr, print_hypothesis_generation
from app.model import common
from app.post_process import (
    ExtractStatus,
    extract_diff_one_instance,
    record_extract_status,
)
from app.task import SweTask, Task

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
You ultimate goal is to figure out the underlying cause of the issue and how to fix it.
"""

# used for strategies: hypothesis, hypothesis-reflect, and prefix for hypothesis-diversify
USER_PROMPT = """Write BRIEF, succinct high-level reasoning for the root cause behind the issue and how to fix it, based on the provided code snippets. 
Write your reason in the format below. Within `<hypothesis></hypothesis>`, replace `...` with the short high-level reasoning for the root cause of the bug and the solution to address it. 

<hypothesis>
...
</hypothesis>
"""


def get_user_prompt(output_dir: str):
    """Return user prompt depending on command line arguments/run configuration"""

    return USER_PROMPT
    


def run(
    message_thread: MessageThread,
    output_dir: str,
    task: Task,
    retries=1,
    print_callback: Callable[[dict], None] | None = None,
):
    """
    Run agent to generate 'hypothesis' AKA high-level explanation of root cause of issue.
    Modified from run_with_retries in agent_write_patch.py
    """

    for i in range(retries):
        
        # (1) replace system prompt
        messages = deepcopy(message_thread.messages)
        new_thread: MessageThread = MessageThread(messages=messages)
        new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

        # (2) add the initial user prompt
        user_prompt = get_user_prompt(output_dir)
        new_thread.add_user(user_prompt)
        print_acr(user_prompt, "Hypothesis Generation", print_callback=print_callback)
        
        result_msg = ""
    
        # debug_file = pjoin(output_dir, f"debug_agent_write_hypothesis_{i+1}.json")
        # with open(debug_file, "w") as f:
        #     json.dump(new_thread.to_msg(), f, indent=4)

        logger.info(f"Trying to write a pre-hypothesis. Try {i+1} of {retries}.")

        raw_hypothesis_file = pjoin(output_dir, f"agent_pre_hypothesis_{i+1}.txt")

        # actually calling model
        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, [])  # no tools

        logger.info(f"Raw pre-hypothesis produced in try {i+1}. Writing into file.")

        with open(raw_hypothesis_file, "w") as f:
            f.write(res_text)

        print_hypothesis_generation(
            res_text, f"try {i+1} / {retries}", print_callback=print_callback
        )

        result_msg = f"Extracted a pre-hypothesis. Consider validating:\n{res_text}"
        # new_thread.add_user(result_msg)  # just for logging
        print_acr(
            result_msg,
            f"hypothesis generation try {i+1} / {retries}",
            print_callback=print_callback,
        )

    return result_msg

