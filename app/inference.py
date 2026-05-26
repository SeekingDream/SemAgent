import os
import inspect
import json
import re
from collections.abc import Callable
from os.path import join as pjoin

from collections import defaultdict
from collections.abc import Iterable
from itertools import cycle
from os import PathLike
from os.path import samefile
from pathlib import Path
from shutil import copy2

from loguru import logger
from termcolor import colored

from app import globals
from app.api.manage import ProjectApiManager
from app.data_structures import FunctionCallIntent, MessageThread
from app.log import (
    log_and_cprint,
    log_and_print,
    log_and_print_acr,
    print_acr,
    print_banner,
    print_issue,
    print_retrieval,
)
from app.model import common, ollama
from app.search.search_manage import SearchManager
from app.utils import parse_function_invocation

from app import config
from app.agents import agent_select
from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reproducer import NoReproductionStep, TestAgent
from app.agents.agent_write_patch import PatchAgent
from app.api import validation
from app.api.review_manage import ReviewManager
from app.api.validation import evaluate_patch
from app.data_structures import BugLocation
from app.log import print_banner, print_issue
from app.manage import ProjectApiManager
from app.model.common import set_model
from app.task import Task

from app.api.call_chain_reviewer import CallChainReviewer

#Missing Natsort
from natsort import natsorted


# FIXME: the system prompt should be different for stratified/state machine.
SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather buggy information, then write patches to solve the issues.
"""

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


def add_step_trigger(orig_prompt: str, is_first: bool = False) -> str:
    """
    Given the original prompt, add the trigger question for the next step.
    Args:
        orig_prompt (str): The original prompt.
        is_first (bool): Whether the trigger is for the first step.
    Returns:
        str: The prompt with trigger question.
    """
    if is_first:
        trigger = "What is the first step?"
    else:
        trigger = "What's the next step to complete the task? Be reminded that you are solving the initial issue."
    return orig_prompt + "\n" + trigger


def start_conversation_round_stratified(
    output_dir: str,
    msg_thread: MessageThread,
    api_manager: ProjectApiManager,
    start_round_no: int = 0,
    print_callback: Callable[[dict], None] | None = None,
) -> bool:
    """
    This version uses json data to process API calls, instead of using the OpenAI function calling.
    Advantage is that multiple API calls can be made in a single round.
    """

    # if enabled, generate pre-hypothesis from issue statement
    if globals.pre_hypothesis:
        print_banner("PRE-HYPOTHESIS GENERATION")
        api_manager.start_new_tool_call_layer()
        api_manager.dispatch_intent(FunctionCallIntent("write_pre_hypothesis", {}, None), msg_thread, print_callback=print_callback)


    prompt = (
        "Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project."
        "\n- search_class(class_name: str): Search for a class in the codebase"
        "\n- search_method_in_file(method_name: str, file_path: str): Search for a method in a given file"
        "\n- search_method_in_class(method_name: str, class_name: str): Search for a method in a given class"
        "\n- search_method(method_name: str): Search for a method in the entire codebase"
        "\n- search_code(code_str: str): Search for a code snippet in the entire codebase"
        "\n- search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file"
        "\n\nNote that you can use multiple search APIs in one round."
        "\n\nNow analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs."
    )
    msg_thread.add_user(prompt)

    round_no = start_round_no

    round_count = range(start_round_no, globals.conv_round_limit + 1)

    try_generate_locs = False
    if globals.disable_patch_generation:
        round_count = range(
            start_round_no, start_round_no + globals.context_generation_limit + 1
        )

    for round_no in round_count:
        api_manager.start_new_tool_call_layer()

        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        msg_thread.save_to_file(conversation_file)

        print_banner(f"CONTEXT RETRIEVAL ROUND {round_no}")

        print_acr(
            prompt,
            f"context retrieval round {start_round_no}",
            print_callback=print_callback,
        )

        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg())
        msg_thread.add_model(res_text, tools=[])
        print_retrieval(res_text, f"round {round_no}", print_callback=print_callback)

        selected_apis, _, proxy_threads = api_manager.proxy_apis(res_text)

        proxy_log = Path(output_dir, f"agent_proxy_{round_no}.json")
        proxy_messages = [thread.to_msg() for thread in proxy_threads]
        proxy_log.write_text(json.dumps(proxy_messages, indent=4))

        if selected_apis is None:
            msg = "The search API calls seem not valid. Please check the arguments you give carefully and try again."
            msg_thread.add_user(msg)
            print_acr(
                msg,
                f"context retrieval round {round_no}",
                print_callback=print_callback,
            )
            continue

        selected_apis_json = json.loads(selected_apis)

        json_api_calls = selected_apis_json.get("API_calls", [])
        buggy_locations = selected_apis_json.get("bug_locations", [])

        formatted = []
        if json_api_calls:
            formatted.append("API calls:")
            for call in json_api_calls:
                formatted.extend([f"\n- `{call}`"])

        if buggy_locations:
            formatted.append("\n\nBug locations")
            for location in buggy_locations:
                s = ", ".join(f"{k}: `{v}`" for k, v in location.items())
                formatted.extend([f"\n- {s}"])
            Path(output_dir, f"fix_locations_{round_no}.json").write_text(
                json.dumps(buggy_locations, indent=4)
            )

        print_acr(
            "\n".join(formatted),
            "Agent-selected API calls",
            print_callback=print_callback,
        )

        # collected enough information to write patch
        if buggy_locations and (not json_api_calls):
            collated_tool_response = "Here is the code in buggy locations:\n\n"
            # provide the buggy locations to the model
            for bug_location in buggy_locations:
                tool_output, *_ = search_for_bug_location(
                    api_manager, msg_thread, bug_location
                )
                collated_tool_response += f"\n\n{tool_output}\n"

            if (
                "Unknown function" not in collated_tool_response
                and "Could not" not in collated_tool_response
            ):
                msg_thread.add_user(collated_tool_response)

                if globals.disable_patch_generation:
                    logger.debug(
                        "Gathered enough information. Skipping patch generation due to feature flag."
                    )
                else:
                    # print_banner("PATCH GENERATION")
                    logger.debug("Gathered enough information. Invoking write_patch.")
                    print_acr(
                        collated_tool_response,
                        "patch generation round 1",
                        print_callback=print_callback,
                    )
                break

            msg = "The buggy locations is not precise. You may need to check whether the arguments are correct and search more information."
            msg_thread.add_user(msg)
            print_acr(
                msg,
                f"context retrieval round {round_no}",
                print_callback=print_callback,
            )
            continue

        # prepare response from tools
        collated_tool_response = ""

        for api_call in json_api_calls:
            func_name, func_args = parse_function_invocation(api_call)

            arg_spec = inspect.getfullargspec(getattr(SearchManager, func_name))
            arg_names = arg_spec.args[1:]  # first parameter is self

            assert len(func_args) == len(
                arg_names
            ), f"Number of argument is wrong in API call: {api_call}"

            kwargs = dict(zip(arg_names, func_args))
            intent = FunctionCallIntent(func_name, kwargs, None)
            tool_output, _, _ = api_manager.dispatch_intent(intent, msg_thread)

            collated_tool_response += f"Result of {api_call}:\n\n"
            collated_tool_response += tool_output + "\n\n"

        msg_thread.add_user(collated_tool_response)
        print_acr(
            collated_tool_response,
            f"context retrieval round {round_no}",
            print_callback=print_callback,
        )

        msg = "Let's analyze collected context first"
        msg_thread.add_user(msg)
        print_acr(
            msg, f"context retrieval round {round_no}", print_callback=print_callback
        )

        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg())
        msg_thread.add_model(res_text, tools=[])
        print_retrieval(res_text, f"round {round_no}", print_callback=print_callback)

        if round_no < globals.conv_round_limit:
            msg = (
                "Based on your analysis, answer below questions:"
                "\n- do we need more context: construct search API calls to get more context of the project. (leave it empty if you don't need more context)"
                "\n- where are bug locations: buggy files and methods. (leave it empty if you don't have enough information)"
            )
            if isinstance(common.SELECTED_MODEL, ollama.OllamaModel):
                # llama models tend to always output search APIs and buggy locations.
                msg += "\n\nNOTE: If you have already identified the bug locations, do not make any search API calls."
            msg_thread.add_user(msg)
            print_acr(
                msg,
                f"context retrieval round {round_no}",
                print_callback=print_callback,
            )
    else:
        log_msg = "Try writing patch anyway."
        # TODO can be improved more
        if globals.disable_patch_generation:
            all_locs = []
            for fix_location_file in Path(output_dir).glob("*fix_locations_*.json"):
                all_locs += json.loads(Path(fix_location_file).read_text())
            all_locs = list(set(map(json.dumps, all_locs)))
            Path(output_dir, "fix_locations.json").write_text(
                json.dumps(all_locs, indent=4)
            )
            try_generate_locs = all_locs != []
            log_msg = "Try outputing some locations still."

        logger.info(f"Too many rounds. {log_msg}")

    round_no += 1

    # HM: atp done with code search, switching to writing patch/fix locations. Add optional detour to first generate hypothesis.
    
    if globals.patch_generation_mode == "hypothesis":
        print_banner("HYPOTHESIS GENERATION")
        # generate hypothesis before patch, giving current conversation/msg thread as context
        api_manager.start_new_tool_call_layer()
        api_manager.dispatch_intent(FunctionCallIntent("write_hypothesis", {}, None), msg_thread, print_callback=print_callback)
    elif globals.patch_generation_mode in ["hypothesis-reflect", "hypothesis-diversify"]:
        print_banner("HYPOTHESIS GENERATION")
        api_manager.start_new_tool_call_layer()
        api_manager.dispatch_intent(FunctionCallIntent("write_hypothesis", {}, None), msg_thread, print_callback=print_callback)
        print_banner("HYPOTHESIS SELECTION")
        api_manager.start_new_tool_call_layer()
        api_manager.dispatch_intent(FunctionCallIntent("select_hypothesis", {}, None), msg_thread, print_callback=print_callback)

    if not globals.disable_patch_generation:
        print_banner("PATCH GENERATION")
        intent = FunctionCallIntent("write_patch", {}, None)
    elif try_generate_locs:
        intent = FunctionCallIntent("propose_locs", {}, None)
    else:
        intent = None

    if intent:
        api_manager.start_new_tool_call_layer()
        api_manager.dispatch_intent(intent, msg_thread, print_callback=print_callback)
        logger.info(f"Invoked {intent.func_name}.")

    logger.info("Ending workflow.")
    conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
    msg_thread.save_to_file(conversation_file)

    return True

def skip_conversation_round(
    output_dir: str,
    msg_thread: MessageThread,
    api_manager: ProjectApiManager,
    print_callback: Callable[[dict], None] | None = None,
) -> bool:

    # find latest conversation file and fix locations file
    convo_files = [x for x in os.listdir(globals.task_input_dir) if x.startswith("conversation_round_")]
    fix_loc_files = [x for x in os.listdir(globals.task_input_dir) if x.startswith("agent_fix_locations_")]
    if not fix_loc_files: fix_loc_files = [x for x in os.listdir(globals.task_input_dir) if x.startswith("fix_locations_")]
    if not convo_files:
        logger.info("\nNo conversation rounds saved for this task!\n")
        print("\nNo conversation rounds saved for this task!\n")
        return False
    if not fix_loc_files:
        logger.info("\nNo suggested fix locations exist. Continuing.\n")
        print("\nNo suggested fix locations exist. Continuing.\n")

    # find most recent convo and fix location file
    numbers = [int(file.split(".")[0].split("_")[-1]) for file in convo_files]
    if fix_loc_files: fix_numbers = [int(file.split(".")[0].split("_")[-1]) for file in fix_loc_files]
    if not numbers:
        logger.info("\nIssue with naming of conversation round json files\n")
        print("\nIssue with naming of conversation round json files\n")
        return False
    if fix_loc_files and not fix_numbers:
        logger.info("\nIssue with naming of agent fix location files. Continuing.\n")
        print("\nIssue with naming of agent fix location files. Continuing.\n")

    msg_thread_load_f = pjoin(globals.task_input_dir, f"conversation_round_{max(numbers)}.json")
    msg_thread_save_f = pjoin(output_dir, f"conversation_round_{max(numbers)}.json")
    if fix_loc_files and fix_numbers: 
        if os.path.isfile(pjoin(output_dir, f"agent_fix_locations_{max(fix_numbers)}.json")):
            fix_file = pjoin(globals.task_input_dir, f"agent_fix_locations_{max(fix_numbers)}.json")
        else:
            fix_file = pjoin(globals.task_input_dir, f"fix_locations_{max(fix_numbers)}.json")
    else: fix_file = None
    
    # load message thread from context retrieval
    msg_thread = MessageThread.load_from_file(msg_thread_load_f)

    # add proposed bug locations if available
    if fix_file:
        with open(fix_file) as f:
            fix_locs = json.load(f)
        message = "The following code locations have been identified as possible buggy locations where modifications may be necessary:\n" + str(fix_locs)
        msg_thread.add_user(message)
    
    # save context to output directory
    msg_thread.save_to_file(msg_thread_save_f)

    # start patch generation
    intent = FunctionCallIntent("write_patch", {}, None)

    api_manager.start_new_tool_call_layer()
    api_manager.dispatch_intent(intent, msg_thread, print_callback=print_callback)
    logger.info(f"Invoked {intent.func_name}.")

    return True


def search_for_bug_location(
    api_manager: ProjectApiManager,
    msg_thread: MessageThread,
    bug_location: dict[str, str],
) -> tuple[str, str, bool]:
    found = False

    file_name = bug_location.get("file")
    method_name = bug_location.get("method")
    class_name = bug_location.get("class")

    assert method_name or class_name, f"Invalid bug location: {bug_location}"

    call_result = None

    def call_function(func_name: str, kwargs: dict[str, str]) -> None:
        nonlocal found, call_result

        intent = FunctionCallIntent(func_name, kwargs, None)
        call_result = api_manager.dispatch_intent(intent, msg_thread)
        _, _, call_is_ok = call_result
        found |= call_is_ok

    if (not found) and method_name and class_name:
        kwargs = {"method_name": method_name, "class_name": class_name}
        call_function("search_method_in_class", kwargs)

    if (not found) and method_name and file_name:
        kwargs = {"method_name": method_name, "file_name": file_name}
        call_function("search_method_in_file", kwargs)

    if (not found) and class_name and file_name:
        kwargs = {"class_name": class_name, "file_name": file_name}
        call_function("search_class_in_file", kwargs)

    if (not found) and class_name:
        kwargs = {"class_name": class_name}
        call_function("get_class_full_snippet", kwargs)

    if (not found) and method_name:
        kwargs = {"method_name": method_name}
        call_function("search_method", kwargs)

    assert call_result

    return call_result


def dump_tool_call_layers_to_file(
    tool_call_layers: list[dict], output_dir: str
) -> None:
    """Dump the layers of tool calls to a file."""
    tool_call_file = pjoin(output_dir, "tool_call_layers.json")
    with open(tool_call_file, "w") as f:
        json.dump(tool_call_layers, f, indent=4)


def start_conversation_round_state_machine(
    output_dir: str,
    msg_thread: MessageThread,
    api_manager: ProjectApiManager,
    start_round_no: int = 0,
) -> bool:
    """
    Start the actual rounds of conversations with model.

    Args:
        output_dir (str): Path to the output directory.
        msg_thread (MessageThread): The message thread to be used.
        api_manager (ProjectApiManager): The API manager to be used.
        start_round_no (int): The round number to start with.
    """
    round_no = start_round_no
    for round_no in range(start_round_no, globals.conv_round_limit + 1):
        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        msg_thread.save_to_file(conversation_file)
        log_and_cprint(
            f"\n========== Conversation Round {round_no} ==========", style="red bold"
        )
        log_and_print(f"{colored('Current message thread:', 'green')}\n{msg_thread}")

        allowed_tools = api_manager.next_tools()
        # TODO: configure the list of tools based on state machine
        tools = ProjectApiManager.get_full_funcs_for_openai(allowed_tools)

        log_and_cprint(f"Current tool state: {api_manager.curr_tool}", style="yellow")
        log_and_cprint(f"Allowed next tool states: {allowed_tools}", style="yellow")

        # create a new iteration of conversation
        res_text, raw_tool_calls, func_call_intents, *_ = common.SELECTED_MODEL.call(
            msg_thread.to_msg(), tools=tools
        )
        log_and_print(
            f"{colored('This roud model response (text):', 'blue')} {res_text}"
        )
        # model can decide whether to create a function call
        if len(func_call_intents) == 1:
            # good case in which we can check function call
            func_call_intent: FunctionCallIntent = func_call_intents[0]
            log_and_print(
                f"{colored('This round model response (function call):', 'blue')} {func_call_intent}"
            )
            # dispatch this function call
            this_model_response = res_text
            this_model_tools = raw_tool_calls
            # add previous call information to user message
            tool_output, summary, _ = api_manager.dispatch_intent(
                func_call_intent, msg_thread
            )
        else:
            # no function call, let's force the model to make one
            this_model_tools = []
            this_model_response = res_text
            tool_output = ""
            summary = "There is no function call in your previous response. Make sure you include one function call. "

        next_user_message = add_step_trigger(summary)

        # form message thread for next round. should include what the model said as well
        msg_thread.add_model(this_model_response, this_model_tools)
        if this_model_tools:
            tool_call_id = this_model_tools[0].id
            msg_thread.add_tool(tool_output, tool_call_id)
            msg_thread.add_user(next_user_message)
        else:
            msg_thread.add_user(next_user_message)

        if len(func_call_intents) == 1:
            func_call_name = func_call_intents[0].func_name
            if func_call_name == "write_patch":
                log_and_print("Ending workflow. write_patch has been invoked.")
                break

        log_and_print("Going to next round ..........")
    else:
        log_and_print("Too many rounds. Try writing patch anyway.")
        write_patch_intent = FunctionCallIntent("write_patch", {}, None)
        api_manager.dispatch_intent(write_patch_intent, msg_thread)

    round_no += 1

    # if we end the workflow normally, there is one more round of conversation to store
    conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
    msg_thread.save_to_file(conversation_file)
    return True

def create_eval_summary_with_call_chain_fixes(patch_content:str, code_changes_to_maintain_consistency_of_issue:str)->str:
    if patch_content:
        return f"""
        Your previous patch: <patch> {patch_content} </patch> may have fixed all the issues unless empty, however another agent has identified some other changes that
        need to be made to fix the issue completely: {code_changes_to_maintain_consistency_of_issue}.
        Your goal is to combine the previous patch with these new changes to generate an aggregate patch that completely resolves 
        the issue.
        """
    else:
        return f"""
        You were unable to generate a patch that solved the issue, however another agent has identified some changes that can be used
        to fix the issue: {code_changes_to_maintain_consistency_of_issue}.
        Your goal is to use these new suggested changes to generate a patch that can resolve the issue.
        """

def write_patch_iterative_with_review(
    task: Task,
    output_dir: str,
    review_manager: ReviewManager,
    retries: int = 3,
    with_patch_content: bool = False,
    reproduced_test_content: str | None = None
) -> tuple[bool, str | None]:
    logger.info("Start generating patches with reviewer")
    patch_gen = review_manager.generator()

    eval_summary = None
    patch_content = None

    for _ in range(retries):
        try:
            patch_handle, patch_content = patch_gen.send(eval_summary)
            logger.info("Reviewer approved patch: {}", patch_handle)
        except StopIteration:
            break

        logger.info("Begin evaluating patch: {}", patch_handle)
        eval_passed, eval_summary = validation.evaluate_patch(
            task, patch_handle, patch_content, output_dir
        )

        eval_passed2 = task.execute_reproducer(reproduced_test_content, patch_content)

        if eval_passed and eval_passed2.returncode==0:

            patch_gen.close()
            logger.info(
                "Patch {} passed evaluation. Ending patch generation", patch_handle
            )
            if with_patch_content:
                return True, patch_content
            else:
                return True

        logger.info("Patch {} failed evaluation", patch_handle)

    if with_patch_content:
        return False, None if patch_content is None else patch_content
    else:
        return False


def write_patch_iterative(
    task: Task,
    output_dir: str,
    review_manager: ReviewManager,
    retries: int = 3, 
    with_patch_content: bool = False,
) -> bool:
    logger.info("Start generating patches without reviewer")

    patch_gen = review_manager.patch_only_generator()
    patch_content = None

    for _ in range(retries):
        try:
            patch_handle, patch_content = patch_gen.send(None)
            logger.info("Generated applicable patch: {}", patch_handle)
        except StopIteration:
            break

        logger.info("Begin evaluating patch: {}", patch_handle)
        eval_passed, _ = validation.evaluate_patch(
            task, patch_handle, patch_content, output_dir
        )

        if eval_passed:
            patch_gen.close()

            logger.info(
                "Patch {} passed evaluation. Ending patch generation", patch_handle
            )
            if with_patch_content:
                return True, patch_content
            else:
                return True

        logger.info("Patch {} failed evaluation", patch_handle)

    if with_patch_content:
        return False, None if patch_content is None else patch_content
    else:
        return False


def run_one_task(task: Task, output_dir: str, model_names: Iterable[str]) -> bool:
    """
    Main entry point to run inference on one task.
    Args:
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
        problem_stmt (str): The original problem statement submitted to the task issue.
    """
    assert model_names

    model_name_cycle = cycle(model_names)

    for idx in range(config.overall_retry_limit):
        model_name = next(model_name_cycle)
        set_model(model_name)

        logger.info("Starting overall retry {} with model {}", idx, model_name)

        out_dir = Path(output_dir, f"output_{idx}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # meta.json is used later by convert_response_to_diff(),
        # so it needs to be copied over
        meta_file = Path(output_dir, "meta.json")
        if meta_file.exists():
            copy2(meta_file, out_dir)

        api_manager = ProjectApiManager(task, str(out_dir))

        if _run_one_task(str(out_dir), api_manager, task.get_issue_statement()):
            logger.info("Overall retry {} succeeded; ending workflow", idx)
            break

        logger.info("Overall retry {} failed; proceeding to next retry", idx)

    logger.info("Starting patch selection")

    selected, details = select_patch(task, output_dir)
    Path(output_dir, "selected_patch.json").write_text(json.dumps(details, indent=4))

    logger.info("Selected patch {}. Reason: {}", selected, details["reason"])

    return True


def select_patch(task: Task, output_dir: str | PathLike) -> tuple[str, dict]:

    patches = natsorted(list(Path(output_dir).glob("**/extracted_patch_*.diff")))

    # TODO: These candidate patches must have been dismissed by reviewer. Maybe an
    # assertion should be added to confirm this.
    candidate_patches = [p for p in patches if may_pass_regression_tests(task, p)]

    agent_comment = None
    thread = None

    for p in candidate_patches[::-1]:
        index = p.with_suffix("").name.rpartition("_")[2]
        reviews = natsorted(
            list(p.parent.glob(f"review_p{index}_t*.json")), reverse=True
        )
        if not reviews:
            continue
        # assert len(reviews) == 1, p

        try:
            if json.loads(reviews[0].read_text())["patch-correct"] == "yes":

                last_patch = natsorted(patches)[-1]
                if not samefile(p, last_patch):
                    logger.info(f"{p} is approved and passes validation, but the last patch was {last_patch}")
                    
                # assert samefile(
                #     p, last_patch
                # ), f"{p} is approved and passes validation, but the last patch was {last_patch}"
                selected_patch = p
                reason = "reviewer-approved"
                break
        except Exception as e:
            continue
    else:
        if len(candidate_patches) > 1:
            content_to_indices = defaultdict(list)
            for idx, p in enumerate(candidate_patches):
                content_to_indices[p.read_text()].append(idx)
            items = sorted(
                content_to_indices.items(),
                key=lambda item: (len(item[1]), -item[1][0]),
                reverse=True,
            )

            # if len(items[0]) > 1:
            if False:
                index = items[0][1][0]
                selected_patch = candidate_patches[index]
                reason = "majority,multiple-pass-regression"
            else:
                try:
                    index, agent_comment, thread = agent_select.run(
                        task.get_issue_statement(),
                        [p.read_text() for p in candidate_patches],
                    )
                    reason = "agent-selected,multiple-pass-regression"
                except Exception:
                    index = -1
                    reason = "agent-error,multiple-pass-regression"
                selected_patch = candidate_patches[index]
        elif len(candidate_patches) == 1:
            selected_patch = candidate_patches[0]
            reason = "no-agent,single-pass-regression"
        else:
            content_to_indices = defaultdict(list)
            for idx, p in enumerate(patches):
                content_to_indices[p.read_text()].append(idx)
            items = sorted(
                content_to_indices.items(),
                key=lambda item: (len(item[1]), -item[1][0]),
                reverse=True,
            )

            # if len(items[0]) > 1:
            if False:
                index = items[0][1][0]
                selected_patch = patches[index]
                reason = "majority,none-pass-regression"
            else:
                try:
                    index, agent_comment, thread = agent_select.run(
                        task.get_issue_statement(), [p.read_text() for p in patches]
                    )
                    reason = "agent-selected,none-pass-regression"
                except Exception:
                    index = -1
                    reason = "agent-error,none-pass-regression"
                selected_patch = patches[index]

    rel_selected_patch = str(selected_patch.relative_to(output_dir))

    result = {
        "selected_patch": rel_selected_patch,
        "reason": reason,
    }

    if agent_comment is not None:
        result["agent_comment"] = agent_comment

    if thread is not None:
        thread.save_to_file(Path(output_dir, "agent_selection.json"))

    return str(selected_patch.relative_to(output_dir)), result


def may_pass_regression_tests(task: Task, patch_file: str | PathLike) -> bool:
    if not config.enable_validation:
        return True

    patch_file = Path(patch_file)

    patch_idx = patch_file.with_suffix("").name.rpartition("_")[2]

    regression_file = patch_file.with_name(f"regression_{patch_idx}.json")
    if regression_file.exists():
        return json.loads(regression_file.read_text())["no_additional_failure"]

    task.reset_project()
    pass_evaluation, _ = evaluate_patch(
        task, patch_idx, patch_file.read_text(), str(patch_file.parent)
    )

    return pass_evaluation


def _run_one_task(
    output_dir: str, api_manager: ProjectApiManager, problem_stmt: str
) -> bool:
    print_banner("Starting SemAgent on the following issue")
    print_issue(problem_stmt)

    test_agent = TestAgent(api_manager.task, output_dir)

    repro_result_map = {}
    repro_stderr = ""
    reproduced = False
    reproduced_test_content = None

    try:
        if config.reproduce_and_review is False:
            raise NoReproductionStep
        
        test_handle, test_content, orig_repro_result = (
            test_agent.write_reproducing_test_without_feedback()
        )
        test_agent.save_test(test_handle)

        coord = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
        repro_result_map[coord] = orig_repro_result

        if orig_repro_result.reproduced:
            #repro_stderr = orig_repro_result.stderr

            #TODO: This is probably not required
            reproduced = True
            reproduced_test_content = test_content
            repro_stderr = f"""
                <stderr>{orig_repro_result.stderr}</stderr>
                <files>{orig_repro_result.get_imp_files_in_a_str()}</files>
            """

            if config.use_reproducer_output_for_localization:
                #Utilizing the test for localization
                localization_repro_result = test_agent.create_new_reproducer_that_shows_localization(reproduced_test_content)
                log_and_print_acr(f"final stack trace determined to show the underlying cause: {str(localization_repro_result)}")
                if localization_repro_result:

                    #TODO: This is probably not required
                    is_stack_trace_useful_for_localization, reasoning_for_why = test_agent.determine_if_stack_trace_is_useful_for_localization(localization_repro_result)
                    log_and_print_acr(f"is the stack trace useful?: {is_stack_trace_useful_for_localization}, \n why?: {reasoning_for_why}")

                    #localization_repro_result = localization_repro_result if is_stack_trace_useful_for_localization else orig_repro_result
                    print_acr(f"final repro result: {str(localization_repro_result)}")
                    repro_stderr = f"""
                        <stderr>{localization_repro_result.stderr}</stderr>
                        <files>{localization_repro_result.get_imp_files_in_a_str()}</files>
                    """

    except NoReproductionStep:
        logger.info(
            "Test agent decides that the issue statement does not contain "
            "reproduction steps; skipping reproducer tracing"
        )
    except InvalidLLMResponse as e:
        logger.warning("Failed to write a reproducer test; skipping reproducer tracing")
        repro_stderr = e.extra_info

    if config.enable_sbfl:
        sbfl_result, *_ = api_manager.fault_localization()
    else:
        sbfl_result = ""

    bug_locs: list[BugLocation]

    #############
    ### Method 1:
    # 1) add to system prompt the given patch, add to search_manager, its used everywhere
    # 2) test if this works?
    #############

    bug_locs, search_msg_thread = api_manager.search_manager.search_iterative(
        api_manager.task, sbfl_result, repro_stderr, reproduced_test_content
    )

    logger.info("Search completed. Bug locations: {}", bug_locs)

    # logger.info("Additional class context code: {}", class_context_code)
    # done with search; dump the tool calls used for recording
    api_manager.search_manager.dump_tool_call_layers_to_file()

    # Write patch
    print_banner("PATCH GENERATION")
    logger.debug("Gathered enough information. Invoking write_patch.")

    review_manager = ReviewManager(
        search_msg_thread,
        bug_locs,
        api_manager.search_manager,
        api_manager.task,
        output_dir,
        test_agent,
        repro_result_map,
    )

    if config.reproduce_and_review and reproduced:
        try:
            generated_patch_passed_regression_tests, patch_content =  write_patch_iterative_with_review(
                api_manager.task, output_dir, review_manager, with_patch_content=True, reproduced_test_content=reproduced_test_content
            )

            #############
            ### Method 2:
            # 1) patch_content = new_patch_content
            # 2) bug_locs = new_bug_locs
            #############

            # If true we use the call chain fixer to further fix the issue
            if config.use_call_chain_fix:
                logger.info(
                    "Invoking call chain fixer."
                )
                call_chain_reviewer = CallChainReviewer(patch_content,bug_locs,api_manager.task)
                code_changes_to_maintain_consistency_of_issue = call_chain_reviewer.fix_inconsistencies_using_call_chains()
                eval_summary_with_call_chain_fixes = f"""Extra context: These are extra fixes given by other software engineers to fix the bug: {create_eval_summary_with_call_chain_fixes(patch_content, code_changes_to_maintain_consistency_of_issue)}, analyze this and figure out how to combine it with and your previously generated patch that fixed the main bulk of issue: "{patch_content}" to resolve the issue. NOTE: If the extra fixes are empty that means no changes need to be made to the final patch."""
                # eval_summary_with_call_chain_fixes = "YOU MUST GENERATE A DIFFERENT PATCH FROM YOUR PREVIOUS PATCH AND IT MUST SOLVE THE ISSUE, BUT THIS PATCH YOU ARE ABOUT TO GENERATE MUST BE DIFFERENT FROM THE PREVIOUS ONE: "
                logger.info(
                    f"Call chain fixer output: {eval_summary_with_call_chain_fixes}"
                )

                # update patching agent with call chain fixes
                review_manager.patch_agent.add_eval_summary_with_call_chain_fixes(eval_summary_with_call_chain_fixes)

                generated_patch_passed_regression_tests =  write_patch_iterative_with_review(
                    api_manager.task, output_dir, review_manager, reproduced_test_content=reproduced_test_content
                )

            return generated_patch_passed_regression_tests
        
        # this exception can arise when writing new reproducers
        except NoReproductionStep:
            pass


    # If no reproduction steps:
    generated_patch_passed_regression_tests, patch_content = write_patch_iterative(
        api_manager.task, output_dir, review_manager, with_patch_content=True,
    )

    if config.use_call_chain_fix:
        logger.info(
            "Invoking call chain fixer."
        )
        call_chain_reviewer = CallChainReviewer(patch_content,bug_locs,api_manager.task)
        code_changes_to_maintain_consistency_of_issue = call_chain_reviewer.fix_inconsistencies_using_call_chains()
        eval_summary_with_call_chain_fixes = f"""Extra context: These are extra fixes given by other software engineers to fix the bug: {create_eval_summary_with_call_chain_fixes(patch_content, code_changes_to_maintain_consistency_of_issue)}, analyze this and figure out how to combine it with and your previously generated patch that fixed the main bulk of issue: "{patch_content}" to resolve the issue. NOTE: If the extra fixes are empty that means no changes need to be made to the final patch."""
        # eval_summary_with_call_chain_fixes = "YOU MUST GENERATE A DIFFERENT PATCH FROM YOUR PREVIOUS PATCH AND IT MUST SOLVE THE ISSUE, BUT THIS PATCH YOU ARE ABOUT TO GENERATE MUST BE DIFFERENT FROM THE PREVIOUS ONE: "
        logger.info(
            f"Call chain fixer output: {eval_summary_with_call_chain_fixes}"
        )

        # update patching agent with call chain fixes
        review_manager.patch_agent.add_eval_summary_with_call_chain_fixes(eval_summary_with_call_chain_fixes)

        generated_patch_passed_regression_tests =  write_patch_iterative(
            api_manager.task, output_dir, review_manager,
        )

    logger.info(
        "Invoked write_patch. Since there is no reproducer, the workflow will be terminated."
    )

    return generated_patch_passed_regression_tests

if __name__ == "__main__":
    from app.raw_tasks import RawSweTask

    config.enable_validation = True

    applicable_path = Path(
        "/media/media0/haifeng/projects/reverse-prompt/acr-plus/experiment/06-13-docker-val-loop-lite-try-2-rand/applicable_patch/"
    )
    task_dirs = list(applicable_path.glob("*"))
    for task_dir in task_dirs:
        meta = json.loads(task_dir.joinpath("meta.json").read_text())
        raw_task = RawSweTask(meta["task_id"], meta["setup_info"], meta["task_info"])
        task = raw_task.to_task()
        selected_patch, reason = select_patch(task, task_dir)

        task_dir.joinpath("selected_patch.json").write_text(
            json.dumps({"selected_patch": selected_patch, "reason": reason}, indent=4)
        )
