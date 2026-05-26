import json
import re
from collections import defaultdict
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias

from loguru import logger
from tenacity import retry, stop_after_attempt

from app.agents.agent_common import InvalidLLMResponse
from app.data_structures import MessageThread, ReproResult
from app.log import print_acr, print_reproducer
from app.model.gpt import common
from app.task import Task

SYSTEM_PROMPT = (
    "You are an experienced software engineer responsible for reproducing given issues."
)
SYSTEM_PROMPT_LOCALIZATION = (
    "You are an experienced software engineer that takes in an issue reproducer and modifies it according to the given instructions."
)
SYSTEM_PROMPT_USEFULNESS_OF_STACK_TRACE = (
    "You are a software engineering expert specializing in debugging and static analysis."
)
INITIAL_REQUEST = (
    "Please try to write a standalone python file `reproducer.py` to reproduce"
    " the issue. Put the file in a code block.\n\n"
    "The file would be put in the root directory of the project and executed"
    " by `python3 reproducer.py`. The script should raise an `AssertionError` when"
    " the issue is present and print a stack trace of the issue. The script should also"
    " exit with code 0 when the issue is fixed.\n\n"
    # Reformat the stacktrace, so that context retrieval agent can
    # get the line numbers right later
    "Please use the following function to print the stack trace, so that the line numbers"
    " of the statements are shown clearly:\n"
    "```\n"
    "def print_stacktrace(e: Exception):\n"
    "    import traceback"
    "    import sys"
    "    tb = traceback.extract_tb(e.__traceback__)\n"
    '    print("Traceback (most recent call last):", file=sys.stderr)\n'
    "    for frame in tb:\n"
    "        line_number = frame.lineno\n"
    '        code_context = frame.line.strip() if frame.line else "Unknown"\n'
    "        print(f'  File \"{frame.filename}\"', file=sys.stderr)\n"
    '        print(f"    {line_number}: {code_context}", file=sys.stderr)\n'
    '    print(f"{e.__class__.__name__}: {e}", file=sys.stderr)\n'
    "```\n"
    # Extra comments to try and fix some issues regarding importing the module
    # " Make sure you do not import the actual repo you are working on, imagine you are writing a test in a repository. For example, do not use 'import astropy' when generating a reproducer for astropy issues"
    # " so you need to only use the code and imports in the repository, if you want to work with "
)


class NoReproductionStep(RuntimeError):
    """Raised when issue statement does not contain steps for reproduction."""

    pass


TestHandle: TypeAlias = str


class TestAgent:
    def __init__(self, task: Task, task_dir: str) -> None:
        self.task = task
        self.task_dir = task_dir

        self._request_idx: int = -1
        self._responses: dict[TestHandle, str] = {}
        self._tests: dict[TestHandle, str] = {}
        self._feedbacks: dict[TestHandle, list[str]] = defaultdict(list)
        self._history: list[TestHandle] = []
        self._non_repro_history: list[TestHandle] = []

    def write_reproducing_test_without_feedback(
        self, retries: int = 5 #number of reproducer rounds
    ) -> tuple[TestHandle, str, ReproResult]:
        return self._write_reproducing_test(num_feedbacks=1, retries=retries)

    def write_reproducing_test_with_feedback(
        self, max_feedbacks: int = 5, retries: int = 5
    ) -> tuple[TestHandle, str, ReproResult]:
        return self._write_reproducing_test(
            num_feedbacks=max_feedbacks, retries=retries
        )

    def add_feedback(self, handle: TestHandle, feedback: str) -> None:
        if handle not in self._tests:
            raise ValueError("patch {} does not exist", handle)

        self._feedbacks[handle].append(feedback)

    def _write_reproducing_test(
        self, num_feedbacks: int, retries: int
    ) -> tuple[TestHandle, str, ReproResult]:
        reproducible, guard_thread = self._issue_has_reproduction_steps(
            self.task.get_issue_statement()
        )
        guard_thread.save_to_file(Path(self.task_dir, "conv_reproducible.json"))

        # if not reproducible:
        #     raise NoReproductionStep

        for idx in range(retries):
            feedback_handles = self._select_feedback_handles(num_feedbacks)

            response, test_content, thread = self._write_test(feedback_handles)
            self._request_idx += 1
            print_reproducer(response)
            Path(self.task_dir, f"test_raw_{self._request_idx}.md").write_text(response)
            thread.save_to_file(
                Path(self.task_dir, f"conv_test_{self._request_idx}.json")
            )

            get_files = True #if idx==retries-1 else False
            repro_result = self.task.execute_reproducer(test_content, get_files = get_files)

            print_acr(str(repro_result))

            if repro_result.reproduced:
                handle = self._register_reproducing_test(response, test_content)
                return handle, test_content, repro_result

            handle = self._register_non_reproducing_test(
                response, test_content, repro_result
            )
            logger.info("registered non reproducing test {}", handle)

        raise InvalidLLMResponse(
            f"Failed to write a reproducing test in {retries} attempts", 
            extra_info = f"""
                        <stderr>{repro_result.stderr}</stderr>
                        <files>{repro_result.get_imp_files_in_a_str()}</files>
                    """ if repro_result else "No execute reproducer output"
            )
    
    # finally:
    #     if orig_repro_result:
            # repro_stderr = f"""
            #         <stderr>{orig_repro_result.stderr}</stderr>
            #         <files>{orig_repro_result.get_imp_files_in_a_str()}</files>
            #     """

    @classmethod
    def _issue_has_reproduction_steps(
        cls, issue_statement: str
    ) -> tuple[bool, MessageThread]:
        prefix_thread = MessageThread()

        prefix_thread.add_system(SYSTEM_PROMPT)

        prefix_thread.add_user(f"Here is an issue:\n\n{issue_statement}")

        key = "has-reproducible-example"
        prefix_thread.add_user(
            "Tell me whether the issue contains a reproducible example. Your"
            " answer should take the following Json format:\n"
            "```\n"
            "{\n"
            f'    "{key}": ...\n'
            "}\n"
            "```\n"
            f'where "{key}" should be either `true` or `false`.'
        )

        @retry(stop=stop_after_attempt(15)) #was 3 before
        def query_and_parse():
            response, *_ = common.SELECTED_MODEL.call(
                prefix_thread.to_msg(), response_format="json_object"
            )

            result = json.loads(response)[key]

            if not isinstance(result, bool):
                raise InvalidLLMResponse(
                    "the LLM output of whether or not the issue was reproducable is not a bool"
                )

            thread = deepcopy(prefix_thread)
            thread.add_model(response)

            return result, thread

        return query_and_parse()

    def _select_feedback_handles(self, max_num_feedbacks: int) -> list[TestHandle]:
        if 0 <= max_num_feedbacks <= len(self._history):
            return self._history[-max_num_feedbacks:]
        elif max_num_feedbacks <= len(self._history) + len(self._non_repro_history):
            num_non_repro = max_num_feedbacks - len(self._history)
            return [
                *self._non_repro_history[-num_non_repro:],
                *self._history,
            ]
        else:
            return [*self._non_repro_history, *self._history]

    def _write_test(
        self, history_handles: list[TestHandle] | None = None
    ) -> tuple[str, str | None, MessageThread]:
        history_handles = history_handles or []

        thread = self._construct_init_thread()
        if any(handle in self._feedbacks for handle in history_handles):
            thread.add_user(INITIAL_REQUEST)
        for handle in history_handles:
            if feedbacks := self._feedbacks.get(handle, []):
                thread.add_model(self._responses[handle], [])
                for feedback in feedbacks:
                    thread.add_user(feedback)
            else:
                logger.warning("test {} does not have a feedback; skipping", handle)
        thread.add_user(INITIAL_REQUEST)

        if not history_handles:
            print_acr(INITIAL_REQUEST)

        response, *_ = common.SELECTED_MODEL.call(thread.to_msg())

        return response, self.convert_response_to_test(response), thread

    def _construct_init_thread(self) -> MessageThread:
        thread = MessageThread()
        thread.add_system(SYSTEM_PROMPT)

        prompt = f"Here is an issue:\n\n{self.task.get_issue_statement()}"
        thread.add_user(prompt)

        return thread

    def _register_reproducing_test(
        self, response: str, test_content: str
    ) -> TestHandle:
        handle = str(self._request_idx)

        assert handle not in self._responses
        assert handle not in self._feedbacks
        assert handle not in self._tests
        assert handle not in self._history

        self._responses[handle] = response
        self._tests[handle] = test_content
        self._history.append(handle)

        return handle

    def _register_non_reproducing_test(
        self, response: str, test_content: str, repro_result: ReproResult
    ) -> TestHandle:
        handle = str(self._request_idx)

        assert handle not in self._responses
        assert handle not in self._feedbacks
        assert handle not in self._tests
        assert handle not in self._non_repro_history

        self._responses[handle] = response
        self._tests[handle] = test_content
        self._non_repro_history.append(handle)
        self._feedbacks[handle].append(self._feedback_from_repro_result(repro_result))

        return handle

    def _feedback_from_repro_result(self, repro_result: ReproResult) -> str:
        return (
            "This test did not reproduce the issue.\n"
            "\n"
            "Either one of three things happened:"
            "\n"
            "1) You DID infact correctly reproduce the core issue, but you did NOT follow the specific guidelines to a) exit with a non-zero exit code when the issue occurs or b) you did not raise an AssertionError."
            "\n"
            f"1.a) The test execution exited with code {repro_result.returncode}.\n"
            "\n"
            f"""which means you {"did exit" if repro_result.returncode!=0 else "did not exit"} with a non zero code!"""
            "\n"
            f"1.b) Your Standard error was: {repro_result.stderr}"
            "\n"
            f"""which means you {"did" if repro_result.returncode!=0 else "did not"} raise an Assertion Error!"""
            "\n"
            "or 2) Your logic to reproduce the issue was correct but you had some syntax errors or other minor issues, in which case you just need to iterate on your reproducer and fix these minor issues"
            "\n"
            "or 3) You did not actually reproduce the issue in which case you need to very clearly understand the issue and attempt to solve it"
            "\n"
            f"Here is your standard output: {repro_result.stdout}\n"
        )

    @classmethod
    def convert_response_to_test(cls, response: str) -> str | None:
        blocks = extract_markdown_code_blocks(response)

        if len(blocks) == 1:
            return blocks[0]
        elif len(blocks) == 2 and blocks[1].strip() == "python3 reproducer.py":
            return blocks[0]
        else:
            return None

    def save_test(self, handle: TestHandle) -> None:
        Path(self.task_dir, f"reproducer_{handle}.py").write_text(self._tests[handle])

    #Localization from reproducer

    def create_prompt_to_convert_reproducer_to_localization(self, reproduced_test_content: str)->str:
        return f"""
            You are given a Python script that acts as a reproducer for a software issue. The current script uses custom AssertionError exceptions and static file inspections (e.g., checking file content or conditions manually), which suppresses the original runtime errors from the underlying framework or library.
            Your goal is to convert it into a version that reveals the complete stack trace from the underlying libraries or frameworks. The new script should:

            1) Remove or replace AssertionError blocks that short-circuit execution when detecting symptoms of the issue.
            2) Allow the original exceptions from the framework (e.g., Django, Flask, SQLAlchemy, etc.) to surface naturally during runtime.
            3) For subprocess calls (e.g., subprocess.run()), do not catch CalledProcessError unless you plan to log full stderr and stdout, and then re-raise the exception or print the full decoded output.
            4) Ensure that any exceptions raised during the execution of the reproducer print a full traceback to stderr, including all internal framework calls, so the root cause can be localized to the actual failing file and line.
            5) Do not suppress exceptions unless you're enhancing visibility (e.g., by logging full tracebacks).
            6) Make the output clear and informative to developers trying to localize or debug the issue.

            The result should be a version of the script where:

            1) The issue is reproduced by actual runtime execution, not static content inspection.
            2) Failures print complete tracebacks from the system/framework.

            Here's my current reproducer script:
            {reproduced_test_content}.

            For the issue:
            {self.task.get_issue_statement()}.

            Please provide a modified version that reveals the complete stack trace from the underlying libraries. 
            Please make sure the output should work as a standalone python file `reproducer.py` that reproduces the issue and reveals the complete tracebacks from the system/framework, and please put the file in a code block.
            """
    
    def add_feedback_of_why_it_failed(self, reason: str, how_to_fix_it: str)->str:
        return f"""
        Your previously generated code fails to reveal the complete stack trace from the underlying libraries because of the following reason:
        {reason}.
        Another agent has given suggestions on how to fix the code:
        {how_to_fix_it}.
        Please fix the generated code so that it follows the initial instructions of revealing the complete stack trace from the underlying libraries.
        Please make sure the output should work as a standalone python file `reproducer.py` that reproduces the issue and reveals the complete tracebacks from the system/framework, and please put the file in a code block.
        """
    
    def does_it_return_stacktrace(self, localization_code, repro_result)->tuple[bool,str,str]:
        prompt_to_test_correctness_of_the_code = f"""
        The following code block was run in an attempt to reproduce the issue from the repository "{self.task.repo_name}" on the issue "{self.task.get_issue_statement()}".

        Code:
        {localization_code}

        It produced the following output:
        {{
            "Stdout": "{repro_result.stdout}",
            "Stderr": "{repro_result.stderr}"
        }}

        You are to determine whether this output indicates that the code both:
        1. **Correctly reproduces the reported issue**, and
        2. **Includes a stack trace that identifies the root cause inside the actual source code of the {self.task.repo_name} repository**, not just user-written files.

        Your job is to analyze the `stderr`, `stdout`, and the code, and answer these 3 questions:

        1. `"returns_stacktrace"` — Does the output include a stack trace showing the error occurring **within the {self.task.repo_name} repository source code**, not just user-defined migrations or project code? Return `true` or `false`.
        2. `"reason"` — If false, explain why the stack trace does not trace into the repo's own source code.
        3. `"how_to_fix_it"` — Suggest specific changes to the reproduction script or setup that would cause the stack trace to reach and reveal the underlying Django repo code where the issue originates (e.g., a bug in migration writer or serializer).

        Your output must be valid JSON in the following format:
        {{
            "returns_stacktrace": true or false,
            "reason": "...",
            "how_to_fix_it": "..."
        }}
        """
        temp_thread = MessageThread()
        temp_thread.add_system(SYSTEM_PROMPT_LOCALIZATION)
        temp_thread.add_user(prompt_to_test_correctness_of_the_code)
        response, *_ = common.SELECTED_MODEL.call(temp_thread.to_msg(), response_format="json_object")
        json_of_response = json.loads(response)
        return bool(json_of_response["returns_stacktrace"]), json_of_response["reason"], json_of_response["how_to_fix_it"]
    
    def make_smaller(self, stderr: str, i:int = 100)->str:
        if stderr is None:
            return "No stderr returned by the file"
        stderr_lines = stderr.splitlines()
        if len(stderr_lines) > 2*i:
            # take first 50 and last 50 lines as stderr can be quite long
            stderr_result = "\n".join(stderr_lines[:i] + ["..."] + stderr_lines[-i:])
        else:
            stderr_result = stderr
        return stderr_result

    def create_new_reproducer_that_shows_localization(self, reproduced_test_content: str)->ReproResult|None:
        thread = MessageThread()
        thread.add_system(SYSTEM_PROMPT_LOCALIZATION)
        thread.add_user(self.create_prompt_to_convert_reproducer_to_localization(reproduced_test_content))

        for _ in range(3):
            response, *_ = common.SELECTED_MODEL.call(thread.to_msg())
            thread.add_model(response, [])
            localization_code = self.convert_response_to_test(response)
            print_acr(localization_code)
            repro_result = self.task.execute_reproducer(localization_code, get_files=True)

            if repro_result is not None:
                #repro_result stderr can be quite long, truncate it
                repro_result.stderr = self.make_smaller(repro_result.stderr)

            print_acr(str(repro_result))
            returns_stacktrace, reason, how_to_fix_it = self.does_it_return_stacktrace(localization_code, repro_result)
            print_acr(f"returns_stacktrace:{returns_stacktrace} \n reason:{reason} \n how_to_fix_it:{how_to_fix_it}")
            if returns_stacktrace:
                return repro_result
            else:
                thread.add_user(self.add_feedback_of_why_it_failed(reason, how_to_fix_it))
        return None
    
    def determine_if_stack_trace_is_useful_for_localization(self, localization_repro_result: ReproResult)->tuple[bool,str]:
        #TODO: self.task.repo_name is only for SWEBench tasks
        prompt_to_determine_if_stack_trace_is_useful_for_localization = f"""
        Given the following information, 
        determine whether the provided stack trace would be **useful** for a bug localization agent trying to find the root cause 
        of a given issue in the given repository.

        An important thing to keep in mind is that the bug localization agent finds context using the file names and methods/
        line numbers for issue localization, which would be useful to have in the stack trace.

        Respond with a JSON object with **only two keys**: 
        1) `"is_stack_trace_useful"` whose value is either `true` or `false` depending on whether the stack trace is useful for bug localization.
        and 2) "reason" where you explain why the stack trace would be useful to a context retrieval agent and what parts of the stack trace would be useful. 

        The Repository Name of the issue is:
        {self.task.repo_name}

        The Issue Description is:
        {self.task.get_issue_statement()}

        The provided Stack Trace that you need to analyze:
        {str(localization_repro_result)}

        Only return a JSON object like this:
        {{
            "is_stack_trace_useful": true or false,
            "reason": provide your reasoning...
        }}
        """
 
        temp_thread = MessageThread()
        temp_thread.add_system(SYSTEM_PROMPT_USEFULNESS_OF_STACK_TRACE)
        temp_thread.add_user(prompt_to_determine_if_stack_trace_is_useful_for_localization)
        response, *_ = common.SELECTED_MODEL.call(temp_thread.to_msg(), response_format="json_object")
        json_of_response = json.loads(response)
        return bool(json_of_response["is_stack_trace_useful"]), json_of_response["reason"]


def generator(
    issue_statement: str,
) -> Generator[tuple[str, MessageThread, bool], str | None, None]:
    prefix_thread = MessageThread()
    prefix_thread.add_system(SYSTEM_PROMPT)

    prompt = f"Here is an issue:\n\n{issue_statement}"
    prefix_thread.add_user(prompt)
    # print_acr(prompt, "reproducer test generation")

    prefix_thread.add_user(INITIAL_REQUEST)
    print_acr(INITIAL_REQUEST, "reproducer test generation")

    threads = []

    index = 1
    thread = deepcopy(prefix_thread)
    while True:
        response, *_ = common.SELECTED_MODEL.call(prefix_thread.to_msg())

        thread.add_model(response, [])
        print_reproducer(response, desc=f"Try {index}")

        index += 1

        threads.append(thread)

        code_blocks = extract_markdown_code_blocks(response)

        if len(code_blocks) != 1:
            _ = yield "", thread, False

            new_prompt = (
                f"Expected 1 code block, got {len(code_blocks)}. Please try again."
            )
        else:
            test_content = code_blocks[0]
            evaluation_msg = yield test_content, thread, True

            assert evaluation_msg is not None

            new_prompt = f"The issue reproduction is incorrect. {evaluation_msg} Please try again."

        thread.add_user(new_prompt)


def extract_markdown_code_blocks(content: str) -> list[str]:
    lines = content.splitlines(keepends=True)

    in_code_block = False
    start_pattern = r"\s*```\w*\s*"
    end_pattern = r"\s*```\s*"

    start, end = -1, -1
    intervals = []

    for idx, line in enumerate(lines):
        if (not in_code_block) and re.match(start_pattern, line):
            in_code_block = True
            start = idx + 1
        elif in_code_block and re.match(end_pattern, line):
            in_code_block = False
            end = idx
            intervals.append((start, end))

    res = ["".join(lines[start:end]) for start, end in intervals]
    return res
