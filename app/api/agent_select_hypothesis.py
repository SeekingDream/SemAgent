"""
An agent, which is only responsible for the select_hypothesis tool call.
"""

import os
import json
import shutil
from collections.abc import Callable, Iterable
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app.api import agent_common
from app.api import validation
from app.data_structures import MessageThread, MethodId
from app.log import print_acr, print_hypothesis_selection
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


def run(
    message_thread: MessageThread,
    output_dir: str,
    task: Task,
    retries=1,
    print_callback: Callable[[dict], None] | None = None,
):
    """
    Run agent to select 'hypothesis' AKA high-level explanation of root cause of issue.
    Modified from run in agent_write_hypothesis.py
    """
        
    # (1) replace system prompt
    messages = deepcopy(message_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)
    new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

    # (2) add the initial user prompt
    user_prompt = get_user_prompt(output_dir)
    new_thread.add_user(user_prompt)
    print_acr(user_prompt, "hypothesis selection", print_callback=print_callback)
    
    result_msg = ""

    debug_file = pjoin(output_dir, f"debug_agent_select_hypothesis.json")
    with open(debug_file, "w") as f:
        json.dump(new_thread.to_msg(), f, indent=4)

    logger.info(f"Selecting a hypothesis.")

    final_hypothesis_file = pjoin(output_dir, f"agent_hypothesis_final.txt")

    # actually calling model
    res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

    new_thread.add_model(res_text, [])  # no tools

    selected_hypothesis = parse_reflection(res_text, output_dir)
    if selected_hypothesis is not None: 
        logger.info(f"Hypothesis selected. Writing into file.")
        result_msg = f"Selected a hypothesis."
        with open(final_hypothesis_file, "w") as f:
            f.write(selected_hypothesis)
    else: 
        logger.info(f"No hypothesis selected. Using first generated hypothesis candidate.")
        result_msg = f"No hypothesis selected."    

    new_thread.add_user(result_msg)  # just for logging
    print_acr(
        result_msg,
        f"hypothesis selection",
        print_callback=print_callback,
    )
    print_hypothesis_selection(
        res_text, print_callback=print_callback
    )

    debug_file = pjoin(output_dir, f"debug_agent_select_hypothesis.json")
    with open(debug_file, "w") as f:
        json.dump(new_thread.to_msg(), f, indent=4)

    return result_msg

def parse_reflection(response, output_dir):
    """Parse model response and return selected hypothesis"""

    if "<selection>" in response:
        selection = response.split("<selection>")[1]
        if "</selection>" in selection: selection = selection.split("</selection>")[0]
        hyp_file = os.path.join(output_dir, f"agent_hypothesis_raw_{selection}.txt")
        if os.path.isfile(hyp_file):
            with open(hyp_file, "r") as f:
                hyp_final = f.read()
                return hyp_final

    return None

def get_user_prompt(output_dir):

    # first get hypothesis candidates
    i = 1
    hypothesis_options = ""
    hypothesis_file = pjoin(output_dir, f"agent_hypothesis_raw_{i}.txt")
    assert os.path.isfile(hypothesis_file), "hypothesis file does not exist"

    while os.path.isfile(hypothesis_file):
        with open(hypothesis_file, "r") as f:
            hyp_cand = f.read()

        if hypothesis_options == "": hypothesis_options = f"**Hypothesis {i}**\n\n{hyp_cand}"
        else: hypothesis_options = f"{hypothesis_options}\n\n**Hypothesis {i}**\n\n{hyp_cand}"
        i += 1
        hypothesis_file = pjoin(output_dir, f"agent_hypothesis_raw_{i}.txt")

    # create user prompt
    important_note = """**IMPORTANT NOTE** - Please return a **SINGLE NUMBER** inside of the <selection> ... </selection> tags. Even if there are more than one hypothesis that are equally the best, please only return a **SINGLE NUMBER** !! Only if there are **NO** hypothesis that plausibly explain the crash/suggest a reasonable solution, then you can output None."""

    user_prompt = f"""A group of developers have analyzed the issue and the code. Each developer in the group has suggested a hypothesis that (1) explains the root cause of the issue and (2) includes a high-level natural language plan that will very likely **RESOLVE** this issue and also **FIX** the inherent problem in the code.

Looking at this list of plausible hypotheses, your job is to diligently follow the below instructions.

**Instructions for this task**
1. You must analyze all the different hypothesis. 
2. Using this analysis you must output which is the best hypothesis based on the below criteria
    A. The hypothesis should include a good explanation of the root cause of the issue.
    B. The hypothesis should include an implementable high-level plan that will **RESOLVE** the issue, **FIX** the inherent problem and **NOT** significantly change the original code functionality.
3. If there are multiple hypothesis that are equally good, then choose the hypothesis that has the simpler high-level plan. You must output a **SINGLE** hypothesis number. You **MUST NOT** return more than one hypothesis number.
4. Only if **ALL** the hypothesis do not explain the crash or suggest a reasonable solution, then you should output None. 

When responding, for each hypothesis, first explain the suggested hypothesis followed by an analysis of the good and bad aspects of the hypothesis. Then finish your answer with your selection of the best hypothesis and the rationale behind the selection.

{important_note}

Please look at the issue, code and list of hypothesis given below.

{hypothesis_options}"""

    return user_prompt

