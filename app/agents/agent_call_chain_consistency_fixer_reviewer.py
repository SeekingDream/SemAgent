from app.log import log_and_print
from app.model import common
from app.data_structures import MessageThread
from app.post_process import ExtractStatus, is_valid_json
import re
import json

SYSTEM_PROMPT_REVIEWER = """
You are an expert software code reviewer who evaluates suggestions made by multiple software engineers attempting to solve a specific issue in a code file.

Your role is to:
- Understand the issue in depth.
- Analyze each suggestion carefully, identifying which are necessary, which are helpful, and which may be incorrect or unnecessary.
- Apply clear and well-reasoned judgment to select the best combination of suggestions to fully and correctly resolve the issue.

You prioritize correctness and consistency in your review. Use precise reasoning when justifying your decisions.
"""

class AgentCallChainConsistencyFixerReviewer:
    def __init__(
            self, 
            issue_stmt: str, 
            generalized_issue_stmt_directions: str, 
            patch_content: str,
    ):
        
        self.issue_stmt: str = issue_stmt
        self.generalized_issue_stmt_directions: str = generalized_issue_stmt_directions
        self.patch_content: str = patch_content
        self.num_json_tries: int = 3

    def extract_suggested_patches(self, agent_call_chain_consistency_fixes)->list[tuple[str,str,str]] | None:

        original_start = "<original>"
        original_end = "</original>"
        patched_start = "<patched>"
        patched_end = "</patched>"
        reason_start = "<reason>"
        reason_end = "</reason>"

        original_pattern = re.compile(f"{original_start}(.*?){original_end}", re.DOTALL)
        patched_pattern = re.compile(f"{patched_start}(.*?){patched_end}", re.DOTALL)
        reason_pattern = re.compile(f"{reason_start}(.*?){reason_end}", re.DOTALL)

        original_matches = original_pattern.findall(agent_call_chain_consistency_fixes)
        patched_matches = patched_pattern.findall(agent_call_chain_consistency_fixes)
        reason_matches = reason_pattern.findall(agent_call_chain_consistency_fixes)

        num_original_matches = len(original_matches)
        num_patched_matches = len(patched_matches)
        num_reason_matches = len(reason_matches)

        if num_original_matches!=num_patched_matches:
            log_and_print(f"Error: Mismatch of matches: {num_original_matches} {num_patched_matches} {num_reason_matches}")

        if  (num_original_matches==num_patched_matches) and (num_original_matches!=num_reason_matches):
            reason_matches = ["No reason provided by agent, you must decide if this suggestion is useful or not."]*num_original_matches

        if num_original_matches==0 or num_patched_matches==0:
            log_and_print("Empty: No suggestions")
            return None

        suggested_changes = list(zip(original_matches, patched_matches, reason_matches))
        return suggested_changes
    
    def create_prompt_to_review_patches(self, patches, file_content)->str:
        return f"""
        You are reviewing and trying to solve the following issue in a code file:

        <issue> {self.issue_stmt} </issue>

        The full content of the file is:

        <file content> {file_content} </file content>

        A number of software engineers have provided suggestions, each aiming to solve the issue along with a starting fix to build on that most likely fixes the bulk of the issue:

        <starting fix> {self.patch_content} </starting fix>

        These suggestions consist of the original code, the patched code, and the reasoning for why this change was suggested.
        Your goal is to filter out unnecessary suggestions while keeping the useful ones.

        Useful suggestions fall into one of these categories:
        - Suggestions ensuring the consistency of the fix throughout the file.
        - Suggestions identifying edge cases that might have been missed by the starting fix.
        - Suggestions fixing code that may have been broken by the starting fix.
        - Suggestions that solve the core issue if the starting patch identifies a wrong fix.

        where as,
        
        Unnecessary suggestions fall into one of these categories:
        - Changes that break existing functionality. Use your best judgement when deciding this.
        - The addition of unnecesary try catch statements or assertions which look good in theory but are unnecesary as they might break unit testing for that particular functionality and could have already been caught else where.
        - If the issue is incredibly simple to fix, then avoid overtly complex suggestions which are prone to break some existing functionality.

        You will be given {len(patches)} suggestions in the format of:
        
        0: <original> ... </original> <patched> ... </patched> <reason> ... </reason>
        1: <original> ... </original> <patched> ... </patched> <reason> ... </reason>
        and so on, with 0: representing the 0th patch, 1: the 1st, etc...

        Your output must be a JSON object in the following format:

        {{
            "0": {{
                "reason": "Explanation of why this suggestion is necessary or not.",
                "required": "Required or Not Required",
            }},
            "1": {{
                "reason": "Explanation of why this suggestion is necessary or not.",
                "required": "Required or Not Required",
            }},
            ...
        }}

        - Each key corresponds to the suggestion ID.
        - The "reason" field must contain a concise, clear justification for why the suggestion is necessery or not.
        - The "required" field should be `Required` if the suggestion is required to solve the issue, or `Not Required` if it is not needed.
        - Always begin with your reasoning in the "reason" field, followed by the decision in the "required" field.

        The suggestions are:

        {self.write_patches_one_by_one(patches)}

        If there are no suggestions then return None.
        """
    
    def write_patches_one_by_one(self, patches)->str:
        output_str = ""
        for i, patch in enumerate(patches):
            output_str += f"""

            {i}: <output> {patch[0]} </output> <patched> {patch[1]} </patched> <reason> {patch[2]} </reason>.

            """
        return output_str
    
    def return_relevant_suggestions(self, output_json_data, patches)->str:
        relevant_suggestions = ""
        log_and_print(f"Reviewer Decisions: {output_json_data}")
        for key in output_json_data.keys():
            patches_key = int(key)
            reasoning_and_decision = output_json_data[key] if type(output_json_data[key])==dict else json.loads(output_json_data[key])
            is_required = reasoning_and_decision["required"]
            is_required_reason = reasoning_and_decision["reason"]
            if "Not" not in is_required and "not" not in is_required:
                relevant_suggestions += f"""

            <output> {patches[patches_key][0]} </output> <patched> {patches[patches_key][1]} </patched> <reason> reason for patch: {patches[patches_key][2]}, reviewer reason why suggestion is important: {is_required_reason} </reason>. 

            """
        log_and_print(f"Final Suggestions: {relevant_suggestions}")
        return relevant_suggestions
    
    def review_fixes_given_by_call_chain_fixer(self, agent_call_chain_consistency_fixes: str, file_content: str) -> str:

        patches = self.extract_suggested_patches(agent_call_chain_consistency_fixes)
        if not patches:
            return "No extra suggestions made by call chain fixer to fix the issue."
        
        prompt_to_review_patches = self.create_prompt_to_review_patches(patches, file_content)

        temp_msg_thread = MessageThread()
        temp_msg_thread.add_system(SYSTEM_PROMPT_REVIEWER)
        temp_msg_thread.add_user(prompt_to_review_patches)

        for tries in range(self.num_json_tries):
            res_text_context, *_ = common.SELECTED_MODEL.call(
                temp_msg_thread.to_msg(), response_format="json_object"
            )

            temp_msg_thread.add_model(res_text_context, [])
            output_json_extract_status, output_json_data = is_valid_json(res_text_context)

            if output_json_extract_status == ExtractStatus.IS_VALID_JSON:
                return self.return_relevant_suggestions(output_json_data, patches)
            else:
                temp_msg_thread.add_user(" JSON extraction failed, please make sure the output is strictly in the JSON format mentioned and try again. ")
        return agent_call_chain_consistency_fixes



    

    


    





