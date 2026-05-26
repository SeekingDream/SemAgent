import re
from typing import List
from app.data_structures import BugLocation, MessageThread
from app.model import common
from app.search import search_utils
from app.task import SweTask, Task
from app.post_process import ExtractStatus, is_valid_json
from app.log import print_acr,log_and_print,log_and_print_acr
from app.agents.agent_call_chain_consistency_fixer_reviewer import AgentCallChainConsistencyFixerReviewer
from utils import extract_text_in_angle_brackets
from app import config

import json
import ast


SYSTEM_PROMPT = """
You are a software developer working on fixing an issue in your code base. Another software developer has looked into this issue and has given 
a fix that solves the main issue or solves it to the best of the developers ability. Now, your goal is to analyze the possible call chains in a flle, go through and analyze them 
and determine if any additional changes need to be made in addition to the main change to maintain consistency in the code base while 
also fixing any edge cases or existing functionality that might have broken.
"""

SYSTEM_PROMPT_NO_REPETITION = """"
You are a software developer working on fixing an issue in your code base. You are collaborating with multiple software engineers 
to fix the issue, and they have combined their fixes and sent it to you. Your goal is to go through these fixes, identify the similar ones 
and aggregate them together. Your goal is to aggregate all the code fixes accordingly.
"""

class AgentCallChainConsistencyFixer:

    def __init__(
              self,
              bug_locs: List[BugLocation],
              patch_content: str | None,
              task: Task,
    ):
         self.num_retries_for_valid_json = 5
         self.bug_locs: List[BugLocation] = bug_locs
         self.task: Task = task
         self.issue_stmt: str = self.task.get_issue_statement()
         self.generalized_issue_stmt_directions: str = self.summarize_patch_for_generality()
         self.patch_content: str = patch_content if patch_content is not None else " 'the AI agent was not able to produce a patch' "

         self.use_cached_results: bool = config.use_cached_results
         self.cached_results_path: str = config.cached_results_path
         self.cached_results_json: dict

    ### getter functions
    def get_issue_stmt(self)->str:
        return self.issue_stmt
    
    def get_generalized_issue_stmt_directions(self)->str:
        return self.generalized_issue_stmt_directions
    
    def get_patch_content(self)->str:
        return self.patch_content

    ### init helper functions
    def summarize_patch_for_generality(self)->str:
        temp_msg_thread = MessageThread()
        temp_msg_thread.add_system(SYSTEM_PROMPT)
        temp_msg_thread.add_user(self.create_prompt_to_generalize_the_issue())
        res_text, *_ = common.SELECTED_MODEL.call(temp_msg_thread.to_msg())
        directions = extract_text_in_angle_brackets("directions", res_text, single_string=True)
        log_and_print("DIRECTIONS: "+directions)
        return directions
    
    def get_file_content(self, bug_loc: BugLocation):
        return search_utils.get_text_in_python_file(bug_loc.abs_file_path)
    
    ### json helper functions

    def read_json_with_tuple_keys(self, path_to_json: str)->dict:
        with open(path_to_json, 'r') as file:
            loaded_dict = json.load(file) 
        # Convert string keys back to tuples if needed
        original_dict = {ast.literal_eval(k): v for k, v in loaded_dict.items()}
        return original_dict
    
    def write_json_with_tuple_keys(self, path_to_json: str):
        with open(path_to_json, 'w') as f:
            # Convert tuple keys to strings before serialization
            serializable_dict = {str(k): v for k, v in self.cached_results_json.items()}
            # Store this in a new JSON
            json.dump(serializable_dict, f, indent=2)

    ### other helper functions

    def convert_json_relevant_code_snippets_to_text(self, output_json)->str:
        json_to_text = ""
        for key in output_json.keys():
            key_value_text = ""
            if type(output_json[key])==str:
                key_value_text = output_json[key]
            elif type(output_json[key]==list):
                for text in output_json[key]:
                    key_value_text+=(str(text) if text is not None else "") + " "
            else:
                key_value_text = str(output_json[key])
            json_to_text+=key + ": " + key_value_text + "; "
        return json_to_text
    
    def are_all_values_in_a_json_a_str_or_a_list(self, output_json)->tuple[bool,str]:
        for key in output_json.keys():
            if type(output_json[key]) != str and type(output_json[key]) != list:
                return False, str(type(output_json[key]))
        return True, ""
    
    def remove_repetitions(self, text:str)->str:
        temp_msg_thread = MessageThread()
        temp_msg_thread.add_system(SYSTEM_PROMPT_NO_REPETITION)
        temp_msg_thread.add_user(self.create_prompt_to_remove_repetitions(text))
        res_text, *_ = common.SELECTED_MODEL.call(temp_msg_thread.to_msg())
        return extract_text_in_angle_brackets("changes",res_text,single_string=True)

    ### prompt generation helper functions

    def create_prompt_to_generalize_the_issue(self):
        return f"""
        The issue might show a very specific example failing or not clearly explain the intent of the developer, 
        your goal is to understand the exact intent of what the issue conveys, and explain
        if this issue is part of a broader issue and if it is generalizable, and explain what the code should reflect to solve this issue as a whole, 
        then summarize everything you've said.
        Finally end it by by giving a paragraph that would direct an AI agent to fix other changes that need to be made in the code base, 
        be as general as possible. The given issue you need to generalize is: "{self.issue_stmt}". The final paragraph that directs an AI agent 
        should be in a <directions> ... </directions> block. 
        """
 
    def create_prompt_to_analyze_a_step(self, flow:str, step:str, json_relevant_code_snippets_in_text: str, give_issue_statement: bool = False): 
        return f"""You are given an execution 
            flow: <flow>" {flow} "</flow> and general directions on how to fix it: <directions>" {self.generalized_issue_stmt_directions} "</directions> 
            {f"""to solve a software issue: <issue>{self.issue_stmt}</issue>""" if give_issue_statement else ""}. in this flow we focus on the step: <step>"{step}"</step>
            .the relevant code snippets are: <codesnippets>"{json_relevant_code_snippets_in_text}"</codesnippets>, and another AI agent has solved the main issue with a patch 
            of: <patch>"{self.patch_content}"</patch>. Your goal is to analyze these code snippets and identify changes that need to be made to maintain consistency 
            with the general directions provided to solve the overall issue. 
            These can be minor or major changes that try to maintain consistency in the file, 
            fix edge cases that might have been missed, 
            or fix modified code that might break important existing functionality. 
            Please be EXTREMELY thorough, go through each line of context given to make 
            sure that you have not missed anything that needs to be changed. You can give your reasoning, however your final changes must be in 
            <changes>...</changes> angle brackets, with the original code, patched code and reasoning, preferably in a <original>...</original> 
            <patched>...</patched> <reason>...</reason> format. If there are no changes to be made then the output must be "No changes" in the angle brackets, 
            i.e <changes>No changes</changes>.
            """


    def create_flows_extraction_prompt(self, bug_loc: BugLocation):
        return f"""You are an AI assistant trying to solve an issue: "{self.issue_stmt}", you have correctly figured out the main issue and fixed it using the 
        patch: "{self.patch_content}" you however miss other minor changes required to maintain the consistency of the fix throughout the file. the context of the 
        file: "{bug_loc.rel_file_path}" is: "{self.get_file_content(bug_loc)}". Your task is to figure out the different flows of execution of the file along with the names of the methods called.
        We will then look at each execution flow and determine if there needs to be other fixes in addition to the main fix to maintain consistency.

        Each flow must be in the following format:
        <flow>
        <step> describe the approach...</step>
        <step>...</step>
        ...
        </flow>

        <flow>
        <step>...</step>
        <step>...</step>
        ...
        </flow>

        ...

        Each flow much be encase in <flow> </flow> and each step must be in <step> </step>, 
        """
    
    def create_get_context_from_step_prompt(self, step:str, file_content:str):
        return f"""You are analyzing a call chain, and are given steps that describe steps in a flow such as `_read_table_qdp()` calls `_get_tables_from_qdp_file()`,
        along with the entire code in the file that the call chain resides in. Your goal is to understand the relevant code snippets that need to be 
        extracted in order to analyze this step and solve an issue. Your output should be a JSON with the key describing the code snippet, and the 
        value being the code in the code snippet. NOTE: If you are giving the code of a method then you must give the ENTIRE code in the method. Also note that the key and values of the JSON must be strings.
        For example, for the `_read_table_qdp()` calls `_get_tables_from_qdp_file()` step, the key should be something like "code in the _get_tables_from_qdp_file() method",
        and the value should be the code in the _get_tables_from_qdp_file method.
        the resulting JSON structure should look like:
        {{
            "code in the _get_tables_from_qdp_file() method": Get all tables from a QDP file.  Parameters .... (rest of the _get_tables_from_qdp_file method code)
        }}
        The file_content is: "{file_content}", and the step in the call chain is: "{step}".
        Your goal is to find the relevant code snippets as mentioned above.
        NOTE: Try to focus the majority at a method level of granularity, however you can also use a class level or expression level of granularity.
        This information will be given to another AI agent who will suggest code changes to fix the bug and maintain consistency throughtout the file.
        """
    
    def create_prompt_to_remove_repetitions(self, text:str)->str:
        return f"""You are given a text that contains multiple code changes either in a <original>...</original> <patched>...</patched> <reason>...</reason> format, 
        or described in natural language. Many of these might be similar to each other, so your goal is to identify all the similar patches, 
        and aggregate them accordingly to remove as many similar/redundent ones as possible. Your output should again be the aggregated code 
        changes in the <original>...</original> <patched>...</patched> <reason>...</reason> format, 
        where <original>...</original> contains the original code, 
        <patched>...</patched> contains the patched code and 
        <reason>...</reason> contains the reasoning behind the suggested code modification. 
        All these changes must be inside a <changes> </changes> block.
        For example:
        "
        #Your Reasoning for aggregations
        <changes> 
            <original>...</original> <patched>...</patched> <reason>...</reason>
        </changes>
        "
        The given text is: <text> "{text}" </text>. These are are from multiple engineers so the code might have similar changes throughout, your goal 
        is to aggregate then and strictly return it in the 
        "
        #Your Reasoning for aggregations
        <changes> 
            <original>...</original> <patched>...</patched> <reason>...</reason>
            ...
        </changes>
        "
        format.
        """

    ### main entry point
    def fix_buggy_locations(self):
        agent_call_chain_consistency_fixer_reviewer = AgentCallChainConsistencyFixerReviewer(self.issue_stmt, self.generalized_issue_stmt_directions, self.patch_content)
        all_final_fixes = ""
        fixed_bugs: str = ""

        for bug_loc in self.bug_locs:
            if self.use_cached_results:

                # load cache to see if key exists
                self.cached_results_json = self.read_json_with_tuple_keys(self.cached_results_path)

                if (self.task.task_id,bug_loc.rel_file_path) in self.cached_results_json.keys():

                    fixed_bugs = self.cached_results_json[(self.task.task_id,bug_loc.rel_file_path)] 
                    log_and_print(f"using a cached fixed_bugs: {fixed_bugs}")

                else:
                    fixed_bugs = self.get_flows_from_a_bug_location_and_suggest_fixes(bug_loc)
                    log_and_print(f"fixed_bugs: {fixed_bugs}")

                    # cache results if something new was added
                    self.cached_results_json = self.read_json_with_tuple_keys(self.cached_results_path)
                    self.cached_results_json[(self.task.task_id,bug_loc.rel_file_path)] = fixed_bugs
                    self.write_json_with_tuple_keys(self.cached_results_path)

                    # cache results if not present
                    log_and_print_acr(f"""New addition to the cache: {(self.task.task_id,bug_loc.rel_file_path)} """)
            else:
                fixed_bugs = self.get_flows_from_a_bug_location_and_suggest_fixes(bug_loc)
                log_and_print(f"fixed_bugs: {fixed_bugs}")

            # review the fixes
            if config.use_reviewer:
                reviewed_fixed_bugs = agent_call_chain_consistency_fixer_reviewer.review_fixes_given_by_call_chain_fixer(fixed_bugs, self.get_file_content(bug_loc))
            else:
                reviewed_fixed_bugs = fixed_bugs

            all_final_fixes += f"""

            Fixes for the file "{bug_loc.rel_file_path}" are {reviewed_fixed_bugs}.
            """

        return all_final_fixes
    
    def get_flows_from_a_bug_location_and_suggest_fixes(self, bug_loc: BugLocation):
        msg_thread = MessageThread()

        msg_thread.add_system(SYSTEM_PROMPT)
        msg_thread.add_user(self.create_flows_extraction_prompt(bug_loc))

        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg())
        flows = extract_text_in_angle_brackets("flow", res_text)

        log_and_print("FLOWS: "+str(flows))

        file_content = self.get_file_content(bug_loc)
        changes_to_fix_all_flows = ""

        for flow in flows:

            print_acr("FLOW: "+str(flow))

            steps = extract_text_in_angle_brackets("step", flow)

            log_and_print("STEPS: "+str(steps))

            changes_to_fix_flow = ""

            for step in steps:

                log_and_print("STEP: "+str(step))

                # aggregate the strings then combine the results
                changes_to_fix_step = self.fix_step(flow, step, file_content)
                log_and_print("changes to fix step: "+str(changes_to_fix_step))

                # aggregating fixes to fix a step
                changes_to_fix_flow += (changes_to_fix_step if changes_to_fix_step is not None else "") + " "

            # remove repetitions in fixing the flow
            changes_to_fix_flow = self.remove_repetitions(changes_to_fix_flow)
            # return the final fixes to the flow
            changes_to_fix_all_flows += changes_to_fix_flow + " "

        # remove repetitions in fixing all the flow
        final_changes_to_fix_all_flows = self.remove_repetitions(changes_to_fix_all_flows)

        print_acr("FINAL CHANGES: "+final_changes_to_fix_all_flows)

        # return the final fixes to fix all the flows
        return final_changes_to_fix_all_flows

    def fix_step(self, flow: str, step:str, file_content:str)->str:
        # prompt to get relevant code snippets
        temp_msg_thread = MessageThread()
        for i in range(self.num_retries_for_valid_json):
            temp_msg_thread.add_system(SYSTEM_PROMPT)
            temp_msg_thread.add_user(self.create_get_context_from_step_prompt(step, file_content))

            res_text_context, *_ = common.SELECTED_MODEL.call(
                temp_msg_thread.to_msg(), response_format="json_object"
            )

            temp_msg_thread.add_model(res_text_context, [])
            output_json_extract_status, output_json_data = is_valid_json(res_text_context)

            if output_json_extract_status == ExtractStatus.IS_VALID_JSON:
                json_values_are_valid, error_type_of_value = self.are_all_values_in_a_json_a_str_or_a_list(output_json_data)

                if json_values_are_valid:

                    json_relevant_code_snippets_in_text = self.convert_json_relevant_code_snippets_to_text(output_json_data)

                    # prompt to use the code snippets and solve the issue
                    prompt_to_analyze_a_step = self.create_prompt_to_analyze_a_step(flow, step, json_relevant_code_snippets_in_text)
                    temp_msg_thread.add_user(prompt_to_analyze_a_step)
                    print_acr("CONTEXT: "+prompt_to_analyze_a_step)

                    res_text_fixing_a_step, *_ = common.SELECTED_MODEL.call(temp_msg_thread.to_msg())
                    temp_msg_thread.add_model(res_text_fixing_a_step, [])
                    print_acr("TO FIX STEP: "+res_text_fixing_a_step)

                    changes_to_fix_the_step = extract_text_in_angle_brackets("changes", res_text_fixing_a_step, single_string=True)

                    if changes_to_fix_the_step=="No changes":
                        return " "
                    else:
                        return changes_to_fix_the_step
                else:
                    temp_msg_thread.add_user(f"JSON extraction failed because some of the JSON values are incorrectly {error_type_of_value}, please make sure the output is strictly in JSON format with string values and try again.")
            else:
                temp_msg_thread.add_user("JSON extraction failed, please make sure the output is strictly in JSON format with string values and try again.")

        return " "






