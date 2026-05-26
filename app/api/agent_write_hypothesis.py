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

USER_PROMPT_ICE = """Write BRIEF, succinct high-level reasoning for the root cause behind the issue, based on the provided code snippets. 
Write your reason in the format below. Within `<hypothesis></hypothesis>`, replace `...` with the short high-level reasoning for the root cause of the bug. 

<hypothesis>
...
</hypothesis>

Given below is an example for the high-level reasoning behind why the code snippet does not print 
the correct sum of the numbers 1 to 10. 

<code>
sum = 0
i = 0
while i < 10:
    print("A dummy loop")
    sum -= i
sum -= 3
print(f"Sum of numbers from 1 to 10 is {sum-1}")
</code>

<hypothesis>
The above snippet is supposed to calculate and print out the sum of all numbers from 1 to 10. However there are two problems 
in this code. The first problem is that the variable "sum" is being decremented inside the while loop when instead it should be 
incremented correctly. The second problem is that there is an erroneous subtraction of the variable "sum" followed by a faulty
printf statement that prints "sum-1" instead of "sum".
</hypothesis>
"""

USER_PROMPT_ICE_V2 = """Write BRIEF, succinct high-level reasoning for the root cause behind the issue and how to fix it, based on the provided code snippets. 
Write your reason in the format below. Within `<hypothesis></hypothesis>`, replace `...` with the short high-level reasoning for the root cause of the bug and the solution to address it. 

<hypothesis>
...
</hypothesis>

Given below is an example for the high-level reasoning and solution behind why the code snippet does not print 
the correct sum of the numbers 1 to 10. 

<code>
sum = 0
i = 0
while i < 10:
    print("A dummy loop")
    sum -= i
sum -= 3
print(f"Sum of numbers from 1 to 10 is {sum-1}")
</code>

<hypothesis>
The above snippet is supposed to calculate and print out the sum of all numbers from 1 to 10. However there are two problems 
in this code. The first problem is that the variable "sum" is being decremented inside the while loop when instead it should be 
incremented correctly. The second problem is that there is an erroneous subtraction of the variable "sum" followed by a faulty
printf statement that prints "sum-1" instead of "sum".
</hypothesis>
"""

# used for strategies: hypothesis, hypothesis-reflect, and prefix for hypothesis-diversify
USER_PROMPT = """Write BRIEF, succinct high-level reasoning for the root cause behind the issue and how to fix it, based on the provided code snippets. 
Write your reason in the format below. Within `<hypothesis></hypothesis>`, replace `...` with the short high-level reasoning for the root cause of the bug and the solution to address it. 

<hypothesis>
...
</hypothesis>
"""

# # used for strategy: hypothesis-diversify
# USER_PROMPT_MEMORY = """Write BRIEF, succinct high-level reasoning for the root cause behind the issue and how to fix it, based on the provided code snippets. 
# Write your reason in the format below. Within `<hypothesis></hypothesis>`, replace `...` with the short high-level reasoning for the root cause of the bug and the solution to address it. 

# <hypothesis>
# ...
# </hypothesis>

# Please note that you have already generated the below list of hypothesis. Please **DO NOT**
# repeat these hypothesis. Think of new reasons or ways to solve the problem.

# Previous Hypothesis BEGIN\n
# """


def get_user_prompt(output_dir: str):
    """Return user prompt depending on command line arguments/run configuration"""

    # if branch: generate hypothesis independently
    if globals.patch_generation_mode in ["hypothesis", "hypothesis-reflect"]: return USER_PROMPT
    
    # elif branch: generate hypothesis different from previous hypotheses
    elif globals.patch_generation_mode == "hypothesis-diversify":
        prompt_prefix = USER_PROMPT     # prompt to generate hypothesis

        # get previous hypotheses to ensure new hypothesis is different
        i = 1
        prev_hyp_f = pjoin(output_dir, f"agent_hypothesis_raw_{i}.txt")
        prev_hyp = []
        while os.path.isfile(prev_hyp_f):
            with open(prev_hyp_f, "r") as f:
                prev_hyp.append(f.read())
            i += 1
            prev_hyp_f = pjoin(output_dir, f"agent_hypothesis_raw_{i}.txt")

        if len(prev_hyp) == 0: prompt_add = ""
        else:   # add previous hypotheses to prompt
            prompt_add = """Please note that you have already generated the below list of hypotheses. Please **DO NOT**
repeat these hypotheses. Think of new reasons or ways to solve the problem.

Previous Hypotheses BEGIN\n
"""
            for hyp in prev_hyp:
                prompt_add = f"{prompt_add}\n====Hypothesis START====\n{hyp}\n====Hypothesis FINISH====\n"

            prompt_add = f"{prompt_add}\nPrevious Hypothesis END\n\nPlease generate **NEW** hypothesis."
        
        return f"{prompt_prefix}\n\n{prompt_add}"


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

    if globals.patch_generation_mode == "hypothesis": retries = 1

    for i in range(retries):

        # Create new copy of messages each time so previous hypothesis generations 
        #   do not influence current hypothesis generation
        
        # (1) replace system prompt
        messages = deepcopy(message_thread.messages)
        new_thread: MessageThread = MessageThread(messages=messages)
        new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

        # (2) add the initial user prompt
        user_prompt = get_user_prompt(output_dir)
        new_thread.add_user(user_prompt)
        print_acr(user_prompt, "Hypothesis Generation", print_callback=print_callback)
        
        result_msg = ""
    
        debug_file = pjoin(output_dir, f"debug_agent_write_hypothesis_{i+1}.json")
        with open(debug_file, "w") as f:
            json.dump(new_thread.to_msg(), f, indent=4)

        logger.info(f"Trying to write a hypothesis. Try {i+1} of {retries}.")

        raw_hypothesis_file = pjoin(output_dir, f"agent_hypothesis_raw_{i+1}.txt")

        # actually calling model
        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, [])  # no tools

        logger.info(f"Raw hypothesis produced in try {i+1}. Writing into file.")

        with open(raw_hypothesis_file, "w") as f:
            f.write(res_text)

        print_hypothesis_generation(
            res_text, f"try {i+1} / {retries}", print_callback=print_callback
        )

        result_msg = f"Extracted a hypothesis. Consider validating hypothesis:\n{res_text}"
        new_thread.add_user(result_msg)  # just for logging
        print_acr(
            result_msg,
            f"hypothesis generation try {i+1} / {retries}",
            print_callback=print_callback,
        )

    return result_msg

