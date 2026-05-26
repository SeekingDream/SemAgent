import json
from collections.abc import Generator
from pathlib import Path

from loguru import logger

from app.agents import agent_reviewer
from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reproducer import TestAgent, TestHandle
from app.agents.agent_reviewer import Review, ReviewDecision
from app.agents.agent_write_patch import PatchAgent, PatchHandle
from app.data_structures import BugLocation, MessageThread, ReproResult
from app.log import print_acr, print_review
from app.search.search_manage import SearchManager
from app.task import SweTask, Task

class ReviewManager:
    def __init__(
        self,
        context_thread: MessageThread,
        bug_locs: list[BugLocation],
        search_manager: SearchManager,
        task: Task,
        output_dir: str,
        test_agent: TestAgent,
        repro_result_map: (
            dict[tuple[PatchHandle, TestHandle], ReproResult] | None
        ) = None,
    ) -> None:
        self.issue_stmt = task.get_issue_statement()
        self.patch_agent = PatchAgent(
            task,
            search_manager,
            self.issue_stmt,
            context_thread,
            bug_locs,
            output_dir,
        )
        # self.test_agent = TestAgent(task, output_dir)
        self.test_agent = test_agent
        self.task: Task = task
        self.repro_result_map: dict[tuple[PatchHandle, TestHandle], ReproResult] = dict(
            repro_result_map or {}
        )
        self.output_dir = output_dir

    def return_repro_result_map(self):
        return self.repro_result_map

    def patch_only_generator(
        self,
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        try:
            while True:
                (
                    patch_handle,
                    patch_content,
                ) = self.patch_agent.write_applicable_patch_without_feedback()
                self.save_patch(patch_handle, patch_content)

                yield patch_handle, patch_content
        except InvalidLLMResponse as e:
            logger.info("Aborting patch-only with exception: {}", str(e))

    def generator(
        self, rounds: int = 5
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        """
        This is the generator when reproducer is available.
        """
        assert isinstance(
            self.task, SweTask
        ), "Only SweTask is supported for reproducer+patch generator."

        try:
            yield from self._generator(rounds)
        except InvalidLLMResponse as e:
            logger.info("Aborting review with exception: {}", str(e))

    def _generator(
        self, rounds: int
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        issue_statement = self.task.get_issue_statement()

        # TODO: fall back to iterative patch generation when reproduction fails
        if not self.test_agent._history:
            (
                test_handle,
                test_content,
                orig_repro_result,
            ) = self.test_agent.write_reproducing_test_without_feedback()
            self.test_agent.save_test(test_handle)
        else:
            try:
                test_handle = self.test_agent._history[-1]
                test_content = self.test_agent._tests[test_handle]
                orig_repro_result = self.repro_result_map[
                    (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
                ]
            except Exception as e:
                logger.info("Starting test mismatch")
                repro_result_map_passing_tests: list[int] = []
                for keys in self.repro_result_map.keys():
                    if keys[0]==PatchAgent.EMPTY_PATCH_HANDLE:
                        repro_result_map_passing_tests.append(int(keys[1]))

                _tests_passing_tests: list[int] = []
                for keys in self.test_agent._tests.keys():
                    _tests_passing_tests.append(int(keys))

                intersection = [x for x in repro_result_map_passing_tests if x in _tests_passing_tests]

                if intersection is not None:
                    test_handle:TestHandle  = str(max(intersection))
                    test_content = self.test_agent._tests[test_handle]
                    orig_repro_result = self.repro_result_map[
                        (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
                    ]
                    logger.info("Test mismatch fixed with max of intersections")

                else:
                    test_handle:TestHandle = str(max(repro_result_map_passing_tests))
                    test_content = self.test_agent._tests[str(max(_tests_passing_tests))]
                    orig_repro_result = self.repro_result_map[
                        (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
                    ]
                    logger.info("Test mismatch fixed")

        coords = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
        self.repro_result_map[coords] = orig_repro_result
        self.save_execution_result(orig_repro_result, *coords)

        # write the first patch
        (
            patch_handle,
            patch_content,
        ) = self.patch_agent.write_applicable_patch_without_feedback()
        self.save_patch(patch_handle, patch_content)

        for _ in range(rounds):
            patched_repro_result = self.task.execute_reproducer(
                test_content, patch_content
            )

            coords = (patch_handle, test_handle)
            self.repro_result_map[coords] = patched_repro_result
            self.save_execution_result(patched_repro_result, *coords)

            review, review_thread = agent_reviewer.run(
                issue_statement,
                test_content,
                patch_content,
                orig_repro_result,
                patched_repro_result,
            )

            print_review(str(review))
            self.save_review(patch_handle, test_handle, review)
            review_thread.save_to_file(
                Path(self.output_dir, f"conv_review_{patch_handle}_{test_handle}.json")
            )

            if review.patch_decision == ReviewDecision.YES:
                evaluation_msg = yield patch_handle, patch_content
                assert evaluation_msg is not None

                print_acr(evaluation_msg, "Patch evaluation")

                if evaluation_msg:
                    self.patch_agent.add_feedback(patch_handle, evaluation_msg)

            if review.patch_decision == ReviewDecision.NO:
                feedback = self.compose_feedback_for_patch_generation(
                    review, test_content
                )
                self.patch_agent.add_feedback(patch_handle, feedback)
                (
                    patch_handle,
                    patch_content,
                ) = self.patch_agent.write_applicable_patch_with_feedback()
                self.save_patch(patch_handle, patch_content)

            if review.test_decision == ReviewDecision.NO:
                feedback = self.compose_feedback_for_test_generation(
                    review, patch_content
                )
                self.test_agent.add_feedback(test_handle, feedback)
                (
                    test_handle,
                    test_content,
                    orig_repro_result,
                ) = self.test_agent.write_reproducing_test_with_feedback()
                self.test_agent.save_test(test_handle)
                coords = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
                self.repro_result_map[coords] = orig_repro_result
                self.save_execution_result(orig_repro_result, *coords)

    @classmethod
    def compose_feedback_for_patch_generation(cls, review: Review, test: str) -> str:
        return (
            f"The previous patch failed a test written by another developer.\n"
            f"Rethink about the code context, reflect, and write another patch.\n"
            f"You can also write the new patch at other locations.\n"
            f"Here is the test file:\n"
            "```\n"
            f"{test}"
            "```\n"
            f"By executing the test file with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.patch_analysis}\n"
            "\n"
            "Therefore, the patch does not correctly resovle the issue.\n"
            "\n"
            "To correct the patch, here is the advice given by another engineer:\n"
            "\n"
            f"{review.patch_advice}"
        )

    @classmethod
    def compose_feedback_for_test_generation(cls, review: Review, patch: str) -> str:
        return (
            f"Here is a patch to the program:\n"
            "```\n"
            f"{patch}"
            "```\n"
            f"By executing your test with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.test_analysis}"
            "\n"
            "Therefore, the test does not correctly reproduce the issue.\n"
            "\n"
            "To correct the test, here is my advice:\n"
            "\n"
            f"{review.test_advice}"
        )

    def save_patch(self, handle: PatchHandle, content: str) -> None:
        Path(self.output_dir, f"extracted_patch_{handle}.diff").write_text(content)

    def save_test(self, handle: TestHandle, content: str) -> None:
        Path(self.output_dir, f"reproducer_{handle}.py").write_text(content)

    def save_review(
        self, patch_handle: PatchHandle, test_handle: TestHandle, review: Review
    ) -> None:
        path = Path(self.output_dir, f"review_p{patch_handle}_t{test_handle}.json")
        path.write_text(json.dumps(review.to_json(), indent=4))

    def save_execution_result(
        self, result: ReproResult, patch_handle: str, test_handle: str
    ) -> None:
        Path(
            self.output_dir, f"execution_{patch_handle}_{test_handle}.json"
        ).write_text(
            json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "triggered": result.reproduced,
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    pass
