import json
import os
import shutil
from collections.abc import Mapping
from os.path import join as pjoin
from pathlib import Path

from app import log
from app.agents import agent_reproducer
from app.analysis import sbfl
from app.analysis.sbfl import NoCoverageData
from app.search.search_manage import SearchManager
from app.agents.agent_select import run_with_eval, NO_PATCH_PRODUCED

# from app.api.python.validation import PythonValidator
from app.task import Task

class ProjectApiManager:
    def __init__(self, task: Task, output_dir: str):
        # for logging of this task instance
        self.task = task

        # where to write our output
        self.output_dir = os.path.abspath(output_dir)

        # build search manager
        self.search_manager = SearchManager(self.task.project_path, self.output_dir)

        # record layered API calls
        self.tool_call_layers: list[list[Mapping]] = []

    ###################################################################
    ########################## API functions ##########################
    ###################################################################

    def fault_localization(self) -> tuple[str, str, bool]:
        """Localize the faulty code snippets by executing test cases.

        Perform fault localization by running the passing and failing test-cases.
        Returns a list of code snippets that are likely to be related to the issue.
        """
        sbfl_result_file = Path(self.output_dir, "sbfl_result.json")
        sbfl_method_result_file = Path(self.output_dir, "sbfl_result_method.json")

        log_file = None
        try:
            test_file_names, ranked_lines, log_file = sbfl.run(self.task)
        except NoCoverageData as e:
            sbfl_result_file.write_text("")
            sbfl_method_result_file.write_text("")

            log_file = e.testing_log_file

            tool_output = "Error in running localization tool"
            summary = tool_output
            return tool_output, summary, False
        finally:
            if log_file is not None:
                shutil.move(log_file, pjoin(self.output_dir, "run_developer_tests.log"))

        ranked_ranges_abs = sbfl.collate_results(ranked_lines, test_file_names)
        ranked_methods_abs = sbfl.map_collated_results_to_methods(ranked_ranges_abs)

        def relativize_filename(tup: tuple) -> tuple:
            file = tup[0]
            relative_file = os.path.relpath(file, self.task.project_path)
            return (relative_file,) + tup[1:]

        ranked_ranges = [relativize_filename(t) for t in ranked_ranges_abs]
        ranked_methods = [relativize_filename(t) for t in ranked_methods_abs]

        sbfl_result_file.write_text(json.dumps(ranked_ranges, indent=4))

        sbfl_method_result_file.write_text(json.dumps(ranked_methods, indent=4))

        log.log_and_print(f"SBFL result (lines): {ranked_ranges}")
        log.log_and_print(f"SBFL result (methods): {ranked_methods}")

        return self._form_sbfl_output(ranked_methods)

    @classmethod
    def _form_sbfl_output(cls, ranked_methods) -> tuple[str, str, bool]:
        if not ranked_methods:
            # empty sbfl results
            tool_output = "Localization could not produce any output."
            summary = tool_output
            return tool_output, summary, False

        if len(ranked_methods) > 5:
            ranked_methods = ranked_methods[:5]

        tool_output = f"Top-{len(ranked_methods)} suspicious methods:\n"
        for idx, (file, class_name, method_name, _) in enumerate(ranked_methods):

            res_str = f"<file>{file}</file>"
            if class_name:
                res_str += f" <class>{class_name}</class>"
            if method_name:
                res_str += f" <func>{method_name}</func>"

            tool_output += f"Suspicious method #{idx + 1}:\n{res_str}\n\n"

        summary = f"Returned top-{len(ranked_methods)} suspicious methods."

        return tool_output, summary, True

    def reproduce(self, retries: int = 5) -> tuple[str, str, bool]:
        reproducer_gen = agent_reproducer.generator(self.task.get_issue_statement())

        test_content = ""
        success = False

        evaluation_msg = None
        messages = []

        for attempt_idx in range(1, retries + 1):
            test_content, thread, run_ok = reproducer_gen.send(evaluation_msg)

            success |= run_ok

            if success:
                break

            evaluation_msg = ""  # TODO: provide true feedback

            msg = thread.to_msg()
            Path(self.output_dir, f"agent_reproducer_raw_{attempt_idx}.md").write_text(
                msg[-1]["content"]
            )
            messages.append(msg)

        Path(self.output_dir, "agent_reproducer.json").write_text(
            json.dumps(messages, indent=4)
        )

        if success:
            summary = "The tool returned a reproducer test"
        else:
            summary = "The tool failed to write a reproducer test"

        return test_content, summary, success
    
    def select_patch(self, map_of_run_description_to_map_of_issue_patch: dict[str, dict[str, str]]):
        map_of_description_to_patch: dict[str, str] = {}
        patches: list[str] = []
        descriptions: list[str] = []

        for description in map_of_run_description_to_map_of_issue_patch.keys():
            map_of_issue_patch = map_of_run_description_to_map_of_issue_patch[description]

            if self.task.task_id in map_of_issue_patch.keys():
                patch = map_of_issue_patch[self.task.task_id]
                map_of_description_to_patch[description] = patch
                patches.append(patch)
            else:
                map_of_description_to_patch[description] = NO_PATCH_PRODUCED
                patches.append(NO_PATCH_PRODUCED)

            descriptions.append(description)

        if len(patches)==0:
            return NO_PATCH_PRODUCED, NO_PATCH_PRODUCED, False
        
        final_description, reason, _ = run_with_eval(self.task, self.task.get_issue_statement(), patches, descriptions)

        extract_status_json = {
                "extract_status": [
                    "APPLICABLE_PATCH",
                ]
            }

        for idx in range(len(descriptions)):
            description = descriptions[idx]
            patch = patches[idx]

            description_dir_name = description.replace(" ","_").lower()
            description_dir_path = os.path.join(self.output_dir, description_dir_name)
            os.mkdir(description_dir_path)

            result_file = Path(description_dir_path, "extracted_patch_0.diff")
            result_file.write_text(patch)

            if description==final_description:
                output_file = os.path.relpath(result_file, self.output_dir)
                extract_status_json_path = Path(description_dir_path, "extract_status.json")
                extract_status_json_path.write_text(json.dumps(extract_status_json))
                shutil.copy2(Path(self.output_dir, "meta.json"), description_dir_path)

        selection_file = Path(self.output_dir, "selected_patch.json")
        selection_json = {
                "selected_patch": output_file,
                "reason": reason
            }
        selection_file.write_text(json.dumps(selection_json))

        os.rmdir(os.path.join(self.output_dir, "search"))

        return final_description, reason, True



