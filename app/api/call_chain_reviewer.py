from app.data_structures import BugLocation
from app.log import print_acr, print_review
from app.task import Task
from app.agents.agent_call_chain_consistency_fixer import AgentCallChainConsistencyFixer
from typing import Set, List


class CallChainReviewer:
    def __init__(
            self,
            patch_content: str | None,
            bug_locs: list[BugLocation],
            task: Task,
    ):
        
        self.task: Task = task
        self.issue_stmt: str = self,task.get_issue_statement()
        self.patch_content: str | None = patch_content
        self.bug_locs: list[BugLocation] = bug_locs
        self.unique_bug_locs: list[BugLocation] = self.get_unique_bug_locations()
    
    def get_unique_bug_locations(self) -> List[BugLocation]:
        locs: Set[str] = set()
        unique_bug_locations: List[BugLocation] = []

        for bug_locations in self.bug_locs:
            if bug_locations.abs_file_path not in locs:
                unique_bug_locations.append(bug_locations)
                locs.add(bug_locations.abs_file_path)
        return unique_bug_locations
    
    def fix_inconsistencies_using_call_chains(self):
        agent_call_chain_consistency_fixer = AgentCallChainConsistencyFixer(self.unique_bug_locs, self.patch_content, self.task)
        return agent_call_chain_consistency_fixer.fix_buggy_locations()

        








    

    

    

        







