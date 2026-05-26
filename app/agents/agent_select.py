import json
from collections import Counter

from tenacity import retry, stop_after_attempt

from app.data_structures import MessageThread
from app.model import common
from app.task import Task

from app import log

#from app.search import search_utils
#search_utils.get_text_in_python_file(bug_loc.abs_file_path)

NO_PATCH_PRODUCED = "No Patch Produced"

SYSTEM_PROMPT = (
    "You are a pull request reviewer. You need to choose the one PR from multiple that"
    " actually will resolve the given issue."
)

num_tries: int = 5
mode_string: str = "with_ranking"

@retry(stop=stop_after_attempt(3))
def run(
    issue_statement: str, patch_contents: list[str]
) -> tuple[int, str, MessageThread]:

    model = common.SELECTED_MODEL

    prefix_thread = MessageThread()
    prefix_thread.add_system(SYSTEM_PROMPT)

    issue_prompt = f"Here is the issue: <issue>{issue_statement}</issue>.\n"
    prefix_thread.add_user(issue_prompt)

    prefix_thread.add_user("First, please analyze the root cause of the issue.")

    response, *_ = model.call(prefix_thread.to_msg(), temperature=1.0)
    prefix_thread.add_model(response)

    prefix_thread.add_user("Analyze how to resolve the issue.")

    response, *_ = model.call(prefix_thread.to_msg())
    prefix_thread.add_model(response)

    prefix_thread.add_user("Here are some patches:")

    for idx, content in enumerate(patch_contents, start=1):
        prefix_thread.add_user(f"Patch {idx}:\n{content}")

    question = (
        "Based on your analysis, "
        "think about which patch best resolves the issue. Tell me the number of"
        " the patch as well as the reason you choose it. Provide your answer in"
        " the following json format:\n"
        "\n"
        "```\n"
        "{\n"
        '    "patch_number": ...,\n'
        '    "reason": "..."'
        "}\n"
        "```\n"
        "where `patch_number` is one of the patch numbers, and reason is a string"
        " stating the reason to your choice."
    )
    prefix_thread.add_user(question)

    prefix_thread.add_user(
        "NOTE: If multiple patches look reasonable, choose the patch that solves the main issue "
        "and tries to maintain the consistency of the fix throughout the file."
    )

    indices = Counter()

    reason = {}

    for _ in range(num_tries):
        response, *_ = model.call(
            prefix_thread.to_msg(), response_format="json_object",
        )
        prefix_thread.add_model(str(response))

        try:
            data = json.loads(response)
            index = int(data["patch_number"]) - 1
            if index in reason.keys():
                reason[index].append(data["reason"])
            else:
                reason[index] = [data["reason"]]
        except Exception:
            prefix_thread.add_user("your previous output was not a JSON, please generate a valid JSON.")
            index = 0

        indices[index] += 1

    index = indices.most_common(1)[0][0]

    if index >= len(patch_contents):
        raise RuntimeError("out-of-bound patch selection by LLM")

    final_reason = ""
    for idx, reasons in enumerate(reason[index]):
        final_reason += f""" \n  Reason {idx+1}: {reasons} \n """

    return index, final_reason, prefix_thread


def get_single_json_output_from_llm(prefix_thread: MessageThread) -> str:
    response, *_ = common.SELECTED_MODEL.call(
        prefix_thread.to_msg(), response_format="json_object",
    )
    start = response.find('{')
    end = response.rfind('}', start)
    if start != -1 and end != -1:
        response = '{'+ response[start + 1:end] +'}'

    return response

# @retry(stop=stop_after_attempt(3))
def run_with_eval(
    task: Task, issue_statement: str, patch_contents: list[str], descriptions: list[str] | None = None
) -> tuple[int, str, MessageThread]:

    model = common.SELECTED_MODEL

    prefix_thread = MessageThread()
    prefix_thread.add_system(SYSTEM_PROMPT)

    issue_prompt = f"Here is the issue: <issue>{issue_statement}</issue>.\n"
    prefix_thread.add_user(issue_prompt)

    prefix_thread.add_user("First, please analyze the root cause of the issue.")

    response, *_ = model.call(prefix_thread.to_msg(), temperature=1.0)
    prefix_thread.add_model(response)

    prefix_thread.add_user("Analyze how to resolve the issue.")

    response, *_ = model.call(prefix_thread.to_msg())
    prefix_thread.add_model(response)

    prefix_thread.add_user("Here are some patches:")

    if mode_string=="with_ranking":
        patches_and_descriptions = []
        for idx in range(len(patch_contents)):
            patch = patch_contents[idx]
            description = descriptions[idx]
            
            try:
                if patch is not NO_PATCH_PRODUCED:
                    tests_passed, msg, log_file, orig_log_file = task.validate(patch)
                else:
                    tests_passed, msg, log_file, orig_log_file = task.validate(task.make_noop_patch(task.project_path))
            except:
                tests_passed, msg = False, "{"+","*20+"}"
            
            if tests_passed:
                patches_and_descriptions.append((0,patch,description))
            else:
                start = msg.find('{')
                end = msg.find('}', start)

                log.log_and_print_acr(f"msg: {msg}")

                if start != -1 and end != -1:
                    result = msg[start + 1:end]
                    split_results = result.split(",")
                    num_failing_tests = len(split_results)

                log.log_and_print_acr(f"result: {result}")
                log.log_and_print_acr(f"num_failing_tests: {num_failing_tests}")

                patches_and_descriptions.append((num_failing_tests,patch,description))

        patch_contents = sorted(patches_and_descriptions)

    else:
        patches_and_descriptions = []
        for idx in range(len(patch_contents)):
            patches_and_descriptions.append((0,patch_contents[idx],descriptions[idx]))

        patch_contents = patches_and_descriptions

    if descriptions is None:
        for idx, content in enumerate(patch_contents, start=1):
            prefix_thread.add_user(f"Patch {idx}:\n{content}")
    else:
        string_of_patches = ""
        for idx in range(len(patch_contents)):

            description = patch_contents[idx][2]
            patch = patch_contents[idx][1]
            num_failed_test_cases = patch_contents[idx][0]

            ## Write patches into a prompt
            if mode_string=="with_description":
                string_of_patches+=f"""\n Patch {idx+1}: \n Description of How Patch Was Generated: {description} \n Patch Content: {patch} \n """
            elif mode_string=="without_description":
                string_of_patches+=f"""\n Patch {idx+1}: \n Patch Content: {patch} \n """
            elif mode_string=="with_ranking":
                if num_failed_test_cases==0:
                    patch_status = "This patch passed all regression test cases"
                else:
                    patch_status = f"This patch failed on {num_failed_test_cases} test cases during regression testing"
                    
                string_of_patches+=f"""\n Patch {idx+1}: \n Description of How Patch Was Generated: {description}  \n Patch Content: {patch} \n Patch Status: {patch_status} \n"""
                
        prefix_thread.add_user(string_of_patches)
        # log.log_and_print(string_of_patches)

    question = (
        "Based on your analysis, "
        "think about which patch best resolves the issue. Tell me the number of"
        " the patch as well as the reason you choose it. Provide your answer in"
        " the following json format:\n"
        "\n"
        "```\n"
        "{\n"
        '    "patch_number": ...,\n'
        '    "reason": "..."'
        "}\n"
        "```\n"
        "where `patch_number` is one of the patch numbers, and reason is a string"
        " stating the reason to your choice."
        "You are given 3 different patches from 3 different approaches, it is very likely that" #TODO: un hardcode this 3 number
        "If a similar solution is proposed a mojority number of times then it is likely to be correct."
        "Also do not think about a solution that looks the best, but one that would realistically fit the best in the code base."
    )
    prefix_thread.add_user(question)

    if descriptions is None:
        prefix_thread.add_user(
            "NOTE: If multiple patches look reasonable, choose the patch that solves the main issue "
            "and tries to maintain the consistency of the fix throughout the file."
        )
    else:
        prefix_thread.add_user(
            """
            NOTE: If multiple patches look reasonable, choose the patch that solves the main issue
            and tries to maintain the consistency of the fix throughout the file.
            """
        )


    use_multiple_llms_to_select_a_patch: bool = False
    judging_llms: list[str] = ["gemini-2.5-pro-preview-05-06", "vertex_ai/claude-sonnet-4@20250514"]

    if use_multiple_llms_to_select_a_patch is False:

        indices = Counter()
        reason = {}

        for _ in range(num_tries):

            response = get_single_json_output_from_llm(prefix_thread)
            # prefix_thread.add_model(str(response))

            try:
                data = json.loads(response)
                index = int(data["patch_number"]) - 1
                if index in reason.keys():
                    reason[index].append(data["reason"])
                else:
                    reason[index] = [data["reason"]]
                indices[index] += 1
            except Exception as e:
                prefix_thread.add_user(
                "your previous output was not a JSON, and could not be extracted by 'data = json.loads(response)', please generate a valid JSON."
                "Provide your answer in"
                " the following json format:\n"
                "\n"
                "```\n"
                "{\n"
                '    "patch_number": ...,\n'
                '    "reason": "..."'
                "}\n"
                "```\n"
                "where `patch_number` is one of the patch numbers, and reason is a string"
                " stating the reason to your choice."
                )

        assert len(indices)!=0, "The LLM did not create a valid json for any issue"
        index = indices.most_common(1)[0][0]

    else:

        num_llms: int = len(judging_llms)
        patch_picked_by_llm: list[int] = []
        reason_for_llm: list[str] = []

        for model_name in judging_llms:

            indices = Counter()
            reason = {}

            common.set_model(model_name)
            model = common.SELECTED_MODEL

            for _ in range(num_tries):

                response = get_single_json_output_from_llm(prefix_thread)
                # prefix_thread.add_model(str(response))

                try:
                    data = json.loads(response)
                    index = int(data["patch_number"]) - 1
                    if index in reason.keys():
                        reason[index].append(data["reason"])
                    else:
                        reason[index] = [data["reason"]]
                    indices[index] += 1
                except Exception as e:
                    prefix_thread.add_user(
                    "your previous output was not a JSON, and could not be extracted by 'data = json.loads(response)', please generate a valid JSON."
                    "Provide your answer in"
                    " the following json format:\n"
                    "\n"
                    "```\n"
                    "{\n"
                    '    "patch_number": ...,\n'
                    '    "reason": "..."'
                    "}\n"
                    "```\n"
                    "where `patch_number` is one of the patch numbers, and reason is a string"
                    " stating the reason to your choice."
                    )

            assert len(indices)!=0, "claude 4 did not create a valid json for any issue"
            index = indices.most_common(1)[0][0]

            final_reason = ""
            for idx, reasons in enumerate(reason[index]):
                final_reason += f""" \n  Reason {idx+1}: {reasons} \n """

            patch_picked_by_llm.append(index)
            reason_for_llm.append(final_reason)

        if num_llms==2:
            if patch_picked_by_llm[0]==patch_picked_by_llm[1]:
                index = patch_picked_by_llm[0]
            else:
                prompt = f"""
                Two LLMs {judging_llms[0]} and {judging_llms[1]} were used to select a patch that best
                solves the given issue:<issue>{issue_statement}</issue>. These LLMs unfortunately have chosen
                two different patches to be the most likely patch, and it is your goal to figure out which reasoning is better
                and ultimately what patch needs to be selected.
                """
                for idx in range(num_llms):
                    patch_id = patch_picked_by_llm[idx]
                    description = patch_contents[patch_id][2]
                    patch = patch_contents[patch_id][1]
                    num_failed_test_cases = patch_contents[patch_id][0]

                    if num_failed_test_cases==0:
                        patch_status = "This patch passed all regression test cases"
                    else:
                        patch_status = f"This patch failed on {num_failed_test_cases} test cases during regression testing"

                    prompt += f"""
                    Patch {patch_id+1} was chosen by LLM {judging_llms[idx]} for the following reason: "{reason_for_llm[idx]}" \n.
                    Here is a description of How Patch Was Generated: {description}  \n Patch Content: {patch}  
                    Patch Status: {patch_status} \n.
                    """
                prefix_thread.add_user(prompt)

                question_2 = (
                    "Please think about which patch best resolves the issue. Tell me the number of"
                    " the patch as well as the reason you choose it. Provide your answer in"
                    " the following json format:\n"
                    "\n"
                    "```\n"
                    "{\n"
                    '    "patch_number": ...,\n'
                    '    "reason": "..."'
                    "}\n"
                    "```\n"
                    "where `patch_number` is one of the patch numbers, and reason is a string"
                    " where reason states the reasoning to your choice."
                )
                prefix_thread.add_user(question_2)

                indices = Counter()
                reason = {}

                for _ in range(1):

                    response = get_single_json_output_from_llm(prefix_thread)
                    # prefix_thread.add_model(str(response))

                    try:
                        data = json.loads(response)
                        index = int(data["patch_number"]) - 1
                        if index in reason.keys():
                            reason[index].append(data["reason"])
                        else:
                            reason[index] = [data["reason"]]
                    except Exception:
                        prefix_thread.add_user("your previous output was not a JSON, please generate a valid JSON.")
                        index = 0

                indices[index] += 1

            index = indices.most_common(1)[0][0]

        else:
            index = Counter(patch_picked_by_llm).most_common(1)[0][0]

    if index >= len(patch_contents):
        raise RuntimeError("out-of-bound patch selection by LLM")

    final_reason = ""
    for idx, reasons in enumerate(reason[index]):
        final_reason += f""" \n  Reason {idx+1}: {reasons} \n """

    final_description = patch_contents[index][2]
    return final_description, final_reason, prefix_thread
