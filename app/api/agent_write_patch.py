"""
An agent, which is only responsible for the write_patch tool call.
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
from app.log import print_acr, print_patch_generation
from app.model import common
from app.post_process import (
    ExtractStatus,
    extract_diff_one_instance,
    record_extract_status,
)
from app.task import SweTask, Task

HYPOTHESIS_DIR = "manual-hypotheses"

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
You ultimate goal is to write a patch that resolves this issue.
"""

PROMPT_TO_ALLOW_ADDITION_OF_NEW_METHODS = """
When making modifications, follow these guidelines:
- If you are modifying existing code, include only the corresponding code snippet.
- If you need to add a new method (function) into a file, you must insert it by modifying the patch for the method immediately preceding where the new method should appear. Do not attempt to create a patch containing just the new method on its own.
  For example, if the file originally looks like this:

class xyz:
def init(...):
...
def func1(...):
...
def func2(...):
...

and you want to insert a new function `funcNew` immediately after `func1`, your patch should be structured as follows:

modification 1
<file>...</file>
<original>
def func1(...):
...
</original>
<patched>
def func1(...):
...
def funcNew(...):
...
</patched>

Do not include any functions that are above or below the target method in the patch.
You may write multiple modifications if needed.
ALWAYS end your answer with a ```
VERY IMPORTANT: The code in the `<original></original>` block must exactly match the code in the file without any extra comments or modifications.

"""

USER_PROMPT_INIT = """Write a patch for the issue, based on the retrieved context.
You can import necessary libraries.
Return the patch in the format below.
ALWAYS start your answer with a ```
Within `<file></file>`, replace `...` with actual file path.
Within `<original></original>`, replace `...` with the original code snippet from the program.
Within `<patched></patched>`, replace `...` with the fixed version of the original code. 
When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.
ALWAYS end your answer with a ```

```
# modification 1
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 2
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 3
...
```
""" + PROMPT_TO_ALLOW_ADDITION_OF_NEW_METHODS



USER_PROMPT_COT = """Write a patch for the issue, based on the retrieved context.
You can import necessary libraries.
Return the patch in the format below.
ALWAYS start your answer with a ```
Within `<reason></reason>`, replace `...` with the reason for the modification.
Within `<file></file>`, replace `...` with actual file path.
Within `<original></original>`, replace `...` with the original code snippet from the program.
Within `<patched></patched>`, replace `...` with the fixed version of the original code. 
When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.
ALWAYS end your answer with a ```

```
# modification 1
<reason>...</reason>
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 2
<reason>...</reason>
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 3
...
```
"""

USER_PROMPT_HYPOTHESIS = """Write a patch for the issue, based on the retrieved context, the issue, and the correct high-level explanation of the issue provided by the given hypothesis.
You can import necessary libraries.
Return the patch in the format below.
ALWAYS start your answer with a ```
Copy and repeat EXACTLY the provided hypothesis which explains the issue in `<hypothesis>...</hypothesis>`
Within `<reason></reason>`, replace `...` with the reason for the modification.
Within `<file></file>`, replace `...` with actual file path.
Within `<original></original>`, replace `...` with the original code snippet from the program.
Within `<patched></patched>`, replace `...` with the fixed version of the original code. 
When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.
ALWAYS end your answer with a ```

```
<hypothesis>
...
</hypothesis>

# modification 1
<reason>...</reason>
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 2
<reason>...</reason>
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 3
...
```
"""

USER_PROMPT_HYPOTHESIS_2 = """Write a patch for the issue, based on the retrieved context and the natural language explanation of the issue.

### Instructions:
1. **Import necessary libraries** as needed.
2. **Analyze the code** to identify the modifications required.
3. **Copy and repeat** the natural language explanation of the issue provided in `<hypothesis>...</hypothesis>` 
4. **Format** the modification list as shown below:
    - Within `<reason></reason>`, replace `...` with the reason for the modification.
    - Within `<file></file>`, replace `...` with actual file path.
    - Within `<original></original>`, replace `...` with the original code snippet from the program.
    - Within `<patched></patched>`, replace `...` with the fixed version of the original code. 
5. **Multiple modifications** may be provided if necessary.
6. When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
7. ALWAYS start and end your answer with a ```

### Format for Modifications:
```
<hypothesis>
...
</hypothesis>

# modification 1
<reason>...</reason>
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 2
<reason>...</reason>
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 3
...
```
"""

def get_user_prompt(task, output_dir):
    """
    Based on global patch_generation_mode, select the correct user prompt to start patch generation.
    """
    if globals.patch_generation_mode == "cot":
        return USER_PROMPT_COT
    elif globals.patch_generation_mode in ["hypothesis", "hypothesis-manual"]:
        task_id = task.task_id
        # print(f"task id: {task_id}")
        if globals.patch_generation_mode == "hypothesis-manual": hyp_file = pjoin(HYPOTHESIS_DIR, f"{task_id}.txt")
        else: hyp_file = pjoin(output_dir, "agent_hypothesis_raw_1.txt")
        with open(hyp_file, "r") as f:
            hyp = f.read()
        return USER_PROMPT_HYPOTHESIS + f"""

<hypothesis>
{hyp}
</hypothesis>
"""
    elif globals.patch_generation_mode in ["hypothesis-reflect", "hypothesis-diversify"]:
        task_id = task.task_id
        # print(f"task id: {task_id}")
        hyp_file = pjoin(output_dir, f"agent_hypothesis_final.txt")
        if not os.path.isfile(hyp_file):
            hyp_file = pjoin(output_dir, "agent_hypothesis_raw_1.txt")
        with open(hyp_file, "r") as f:
            hyp = f.read()
        return USER_PROMPT_HYPOTHESIS + f"""

{hyp}
"""
    else:
        return USER_PROMPT_INIT

def run_with_retries(
    message_thread: MessageThread,
    output_dir: str,
    task: Task,
    retries=5,
    print_callback: Callable[[dict], None] | None = None,
) -> tuple[str, float, int, int]:
    """
    Since the agent may not always write an applicable patch, we allow for retries.
    This is a wrapper around the actual run.
    """
    # (1) replace system prompt
    messages = deepcopy(message_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)
    new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

    # (2) add the initial user prompt
    user_prompt = get_user_prompt(task, output_dir)
    new_thread.add_user(user_prompt)
    print_acr(user_prompt, "patch generation", print_callback=print_callback)

    can_stop = False
    result_msg = ""

    for i in range(1, retries + 2):
        if i > 1:
            debug_file = pjoin(output_dir, f"debug_agent_write_patch_{i - 1}.json")
            with open(debug_file, "w") as f:
                json.dump(new_thread.to_msg(), f, indent=4)

        if can_stop or i > retries:
            break

        logger.info(f"Trying to write a patch. Try {i} of {retries}.")

        raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{i}")

        # actually calling model
        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, [])  # no tools

        logger.info(f"Raw patch produced in try {i}. Writing patch into file.")

        with open(raw_patch_file, "w") as f:
            f.write(res_text)

        print_patch_generation(
            res_text, f"try {i} / {retries}", print_callback=print_callback
        )

        # Attemp to extract a real patch from the raw patch
        diff_file = pjoin(output_dir, f"extracted_patch_{i}.diff")
        extract_status, extract_msg = extract_diff_one_instance(
            raw_patch_file, diff_file
        )

        # record the extract status. This is for classifying the task at the end of workflow
        record_extract_status(output_dir, extract_status)

        if extract_status == ExtractStatus.APPLICABLE_PATCH:
            patch_content = Path(diff_file).read_text()
            print_acr(
                f"```diff\n{patch_content}\n```",
                "extracted patch",
                print_callback=print_callback,
            )

            # patch generated is applicable and all edits are ok, so we can think about validation
            if globals.enable_validation:
                # if we have a patch extracted, apply it and validate

                patch_is_correct, err_message, log_file = task.validate(diff_file)
                shutil.move(log_file, pjoin(output_dir, f"run_test_suite_{i}.log"))

                if patch_is_correct:
                    result_msg = (
                        "Written a patch that resolves the issue. Congratulations!"
                    )
                    new_thread.add_user(result_msg)  # just for logging
                    print_acr(
                        result_msg,
                        f"patch generation try {i} / {retries}",
                        print_callback=print_callback,
                    )
                    can_stop = True
                # the following two branches cannot be swapped, because
                # --enable-perfect-angelic is meant to override --enable-angelic
                elif globals.enable_perfect_angelic:
                    if not isinstance(task, SweTask):
                        raise NotImplementedError(
                            f"Angelic debugging not implemented for {type(task).__name__}"
                        )

                    msg = (
                        f"Written an applicable patch, but it did not resolve the issue. Error message: {err_message}.",
                    )

                    incorrect_locations = validation.perfect_angelic_debug(
                        task.task_id, diff_file, task.project_path
                    )
                    angelic_msg = angelic_debugging_message(incorrect_locations[0])

                    result_msg = f"{msg}\n{angelic_msg}"
                    new_thread.add_user(result_msg)
                    print_acr(
                        result_msg,
                        f"patch generation try {i} / {retries}",
                        print_callback=print_callback,
                    )
                    continue
                elif globals.enable_angelic:
                    raise NotImplementedError(
                        "Angelic debugging has not been integrated"
                    )
                else:
                    result_msg = f"Written an applicable patch, but it did not resolve the issue. {err_message} "
                    result_msg += " Please try again."
                    new_thread.add_user(result_msg)
                    print_acr(
                        result_msg,
                        f"patch generation try {i} / {retries}",
                        print_callback=print_callback,
                    )
                    continue
            elif globals.enable_perfect_angelic:
                if not isinstance(task, SweTask):
                    raise NotImplementedError(
                        f"Angelic debugging not implemented for {type(task).__name__}"
                    )

                incorrect_locations = validation.perfect_angelic_debug(
                    task.task_id, diff_file, task.project_path
                )

                msg = "Extracted a patch."
                if angelic_msg := angelic_debugging_message(incorrect_locations[0]):
                    result_msg = f"{msg}\n{angelic_msg}"
                else:
                    result_msg = msg

                new_thread.add_user(result_msg)
                print_acr(
                    result_msg,
                    f"patch generation try {i} / {retries}",
                    print_callback=print_callback,
                )
                continue
            elif globals.enable_angelic:
                raise NotImplementedError("Angelic debugging has not been integrated")
            else:
                result_msg = "Extracted a patch. Since validation is disabled, you should validation the patch later on. Ending the workflow."
                new_thread.add_user(result_msg)  # just for logging
                print_acr(
                    result_msg,
                    f"patch generation try {i} / {retries}",
                    print_callback=print_callback,
                )
                can_stop = True

        else:
            # we dont have a valid patch
            new_prompt = (
                "Your edit could not be applied to the program. "
                + extract_msg
                + " Please try again."
            )
            new_thread.add_user(new_prompt)
            print_acr(
                new_prompt,
                f"patch generation try {i} / {retries}",
                print_callback=print_callback,
            )
            result_msg = "Failed to write a valid patch."

    return result_msg


def angelic_debugging_message(
    incorrect_locations: Iterable[tuple[str, MethodId]],
) -> str:
    msg = []

    if incorrect_locations:
        msg.append("The following methods should not have been changed:")
        msg.extend(
            f"    {filename}: {method_id!s}"
            for filename, method_id in incorrect_locations
        )

    return "\n".join(msg)
