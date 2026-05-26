"""
This agent selects the search APIs to use, and returns the selected APIs in its response in
non-json format.
"""

import re
from collections.abc import Generator

from loguru import logger

from app import config
from app.data_structures import MessageThread
from app.log import print_acr, print_retrieval, log_and_print_acr
from app.model import common, ollama
from app.utils import extract_text_in_angle_brackets
import json

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather sufficient code context for resolving the issue.
The collected context will later be sent to your colleage for writing a patch.
Do not worry about test files or writing test; you are only interested in crafting a patch.
"""


SELECT_PROMPT = (
    "Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project."
    "\n- search_class(class_name: str): Search for a class in the codebase."
    "\n- search_class_in_file(self, class_name, file_name: str): Search for a class in a given file."
    "\n- search_method_in_file(method_name: str, file_path: str): Search for a method in a given file.."
    "\n- search_method_in_class(method_name: str, class_name: str): Search for a method in a given class."
    "\n- search_method(method_name: str): Search for a method in the entire codebase."
    "\n- search_code(code_str: str): Search for a code snippet in the entire codebase."
    "\n- search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file."
    "\n- get_code_around_line(file_path: str, line_number: int, window_size: int): Get the code around a given line number in a file. window_size is the number of lines before and after the line number."
    "\n\nYou must give correct number of arguments when invoking API calls."
    "\n\nNote that you can use multiple search APIs in one round."
    "\n\nNow analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs."
    "\nYou will be given as many api calls as you need to find relevant code and figure out what is going on."
    "\nPlease be thorough in getting the code, rationalizing it properly and then making a very clear cut decision as to where the bug is"
    "\nIt might be useful to search for the code in the suggestions by the external tools, but make sure you know what the code contains and rationalize it before you make a decision"
    "\nDO NOT JUST JUMP TO CONCLUSIONS."
)


ANALYZE_PROMPT = (
    "Let's analyze collected context first.\n"
    "If an API call could not find any code, you should think about what other API calls you can make to get more context.\n"
    "If an API call returns some result, you should analyze the result and think about these questions:\n"
    "1. What does this part of the code do?\n"
    "2. What is the relationship between this part of the code and the bug?\n"
    "3. Given the issue description, what would be the intended behavior of this part of the code?\n"
)


ANALYZE_AND_SELECT_PROMPT = (
    "Based on your analysis, answer below questions:\n"
    "1. do we need more context: construct search API calls to get more context of the project. If you don't need more context, LEAVE THIS EMTPY.\n"
    "2. where are bug locations: buggy files, classes, and methods. "
    "Only include the necessary locations that need to be fixed. "
    "For each bug location, you should also describe the intended behavior of the code at that location, "
    "in order to resolve the issue. "
    "The intended behavior should preserve the existing functionality of the code. "
    "If you think two methods in the same class are buggy, you should provide two bug locations (E.g. Location 1: File: a.py, Class: A, Method: a. Location 2: File: a.py, Class: A, Method: b.). "
    "If you want to add a method to a class, you should only provide the file and class as bug location, "
    "and describe the new method in intended behavior. "
    "If you still need more context, LEAVE THIS EMPTY."
)


# TODO: move this to some util class, since other agents may need it as well
def prepare_issue_prompt(problem_stmt: str) -> str:
    """
    Given the raw problem statement, sanitize it and prepare the issue prompt.
    Args:
        problem_stmt (str): The raw problem statement.
            Assumption: the problem statement is the content of a markdown file.
    Returns:
        str: The issue prompt.
    """
    # remove markdown comments
    problem_wo_comments = re.sub(r"<!--.*?-->", "", problem_stmt, flags=re.DOTALL)
    content_lines = problem_wo_comments.split("\n")
    # remove spaces and empty lines
    content_lines = [x.strip() for x in content_lines]
    content_lines = [x for x in content_lines if x != ""]
    problem_stripped = "\n".join(content_lines)
    # add tags
    result = "<issue>" + problem_stripped + "\n</issue>"
    return result


def generator(
    issue_stmt: str, sbfl_result: str, reproducer_result: str, task_name: str | None = None
) -> Generator[tuple[str, MessageThread], tuple[str, bool] | None, None]:
    """
    Args:
        - issue_stmt: problem statement
        - sbfl_result: result after running sbfl
    """

    msg_thread = MessageThread()

    system_prompt: str = SYSTEM_PROMPT
    if config.is_initial_patch_given is True:
        with open(config.path_to_patches, 'r') as f:
            data = json.load(f)
        init_patch: str = "No Golden Patch Found, Proceed as normal"
        if task_name in data.keys():
            init_patch = data[task_name] 
        add_init_patch_to_prompt = f"""
        The golden patch git diff is given to you, you must ONLY search for the file names in this patch only!
        This is a very strict requirement.
        The golden patch that resolves the issue is:
        <golden patch>
        {init_patch}
        </golden patch>
        This patch is 100% correct guaranteed! Do not try to contradict it!
        """
        system_prompt+=add_init_patch_to_prompt

    msg_thread.add_system(system_prompt)

    issue_prompt = prepare_issue_prompt(issue_stmt)
    msg_thread.add_user(issue_prompt)

    if config.reproduce_and_review and reproducer_result:
        suspicious_files = None
        if "<stderr>" in reproducer_result and "<files>" in reproducer_result:
            suspicious_files = extract_text_in_angle_brackets("files", reproducer_result, True)
            reproducer_result = extract_text_in_angle_brackets("stderr", reproducer_result, True)

        reproducer_prompt = """An external analysis tool has been deployed to construct a standalone reproducer .py file
        that tries to reproduce the issue. It may or may not perform well, but it is up to you
        to choose if you want to use the results from this tool, if you think they are useful."""
        reproducer_prompt += "The tool output is as follows:\n"

        stderr_lines = reproducer_result.splitlines()
        i = 100
        if len(stderr_lines) > 2*i:
            # take first 50 and last 50 lines as stderr can be quite long
            stderr_result = "\n".join(stderr_lines[:i] + ["..."] + stderr_lines[-i:])
        else:
            stderr_result = reproducer_result

        reproducer_prompt += stderr_result
        msg_thread.add_user(reproducer_prompt)

        if suspicious_files:
            msg_thread.add_user(suspicious_files)
            log_and_print_acr(f"suspicious_files: {suspicious_files}")

    if config.enable_sbfl:
        sbfl_prompt = """An external analysis tool has been deployed to identify the suspicious code to be fixed. 
        This tool consists of the suspicious methods found during search based fault localization (SBFL) after the
        addition of tests that actually test what the issue describes. You should probably use the results from
        this tool as they would help in localizing the issue."""
        sbfl_prompt += "The tool output is as follows:\n"
        sbfl_prompt += sbfl_result
        msg_thread.add_user(sbfl_prompt)
        log_and_print_acr(f"sbfl_prompt: {sbfl_prompt}")

    msg_thread.add_user(SELECT_PROMPT)

    print_acr(SELECT_PROMPT, "context retrieval initial prompt")

    # TODO: figure out what should be printed to console here
    # print_acr(prompt, f"context retrieval round {start_round_no}")

    while True:

        # first call is to select some APIs to call
        logger.debug("<Agent search> Selecting APIs to call.")
        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg())
        msg_thread.add_model(res_text)
        # TODO: print the response
        print_retrieval(res_text, "Model response (API selection)")

        # the search result should be sent here by our backend AST search tool
        generator_input = yield res_text, msg_thread
        assert generator_input is not None
        search_result, re_search = generator_input

        if re_search:
            # the search APIs selected have some issue
            logger.debug(
                "<Agent search> Downstream could not consume our last response. Will retry."
            )
            msg_thread.add_user(search_result)
            continue

        # the search APIs selected are ok and the results are back
        # second call is to analyze the search results
        logger.debug("<Agent search> Analyzing search results.")
        msg_thread.add_user(search_result)
        msg_thread.add_user(ANALYZE_PROMPT)
        print_acr(ANALYZE_PROMPT, "context retrieval analyze prompt")

        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg())
        msg_thread.add_model(res_text)
        print_retrieval(res_text, "Model response (context analysis)")

        analyze_and_select_prompt = ANALYZE_AND_SELECT_PROMPT
        if isinstance(common.SELECTED_MODEL, ollama.OllamaModel):
            # llama models tend to always output search APIs and buggy locations.
            analyze_and_select_prompt += "\n\nNOTE: If you have already identified the bug locations, do not make any search API calls."

        msg_thread.add_user(analyze_and_select_prompt)
        print_acr(
            analyze_and_select_prompt, "context retrieval analyze and select prompt"
        )
