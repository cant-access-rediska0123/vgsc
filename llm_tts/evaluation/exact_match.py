import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .grader import get_timeout_count, math_equal
from .parser import STRIP_EXCEPTIONS, extract_answer, strip_string

log = logging.getLogger(__name__)


def _normalize_gold_answer(gold_answer: str, data_name: str) -> str:
    if not gold_answer:
        return ""

    if data_name not in STRIP_EXCEPTIONS:
        return strip_string(gold_answer, skip_unit=data_name == "carp_en")
    else:
        return (
            gold_answer.replace("\\neq", "\\ne")
            .replace("\\leq", "\\le")
            .replace("\\geq", "\\ge")
        )


def _extract_boolean_answer(text: str) -> str | None:
    if not text:
        return None
    text = _strip_thinking_tags(text)
    text_lower = text.lower().strip()
    if re.search(r"\btrue\b", text_lower) or re.search(r"\byes\b", text_lower):
        return "True"
    if re.search(r"\bfalse\b", text_lower) or re.search(r"\bno\b", text_lower):
        return "False"
    return None


def _strip_thinking_tags(text: str) -> str:
    parts = text.rsplit("</think>", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    return re.sub(r"</?think>", "", text).strip()


def _extract_single_letter_answer(text: str) -> str | None:
    if not text:
        return None
    text = _strip_thinking_tags(text)

    boxed_matches = re.findall(r"\\boxed\{([A-Za-z])\}", text)
    if boxed_matches:
        return boxed_matches[-1].upper()

    text_matches = re.findall(r"\\text\{([A-Za-z])\}", text)
    if text_matches:
        return text_matches[-1].upper()

    match = re.search(r"\b([A-Z])[.,]?\s*$", text)
    if match:
        return match.group(1).upper()

    return None


class EvaluatorExactMatch:
    def __init__(
        self,
        dataset_answer_format: str = "numeric",
        data_name: str = None,
        silent: bool = False,
    ):
        """
        Initialize the exact match evaluator.

        Args:
            dataset_answer_format: "numeric", "boolean", "char", or "string"
            data_name: Dataset name (e.g., "minerva_math", "math500", "gsm8k") - REQUIRED
            n_threads: Number of worker threads to use in __call__. Use 1 for serial execution.
        """
        if not data_name:
            raise ValueError("data_name is required for EvaluatorExactMatch")
        self.dataset_answer_format = dataset_answer_format.lower()
        self.data_name = data_name
        self.silent = silent

        if self.dataset_answer_format not in ["numeric", "boolean", "char", "string"]:
            raise ValueError(
                f"dataset_answer_format must be 'numeric', 'boolean', 'char', or 'string', got '{dataset_answer_format}'"
            )

    def _score_single(
        self, inp: tuple[str, str, str], pre_extracted: bool = False
    ) -> float:
        """
        Score a single sample.

        Args:
            inp: (problem, solution, gold_answer) tuple
            pre_extracted: If True, solution is already an extracted answer
                (e.g. from \\boxed{}), so skip extract_answer and only normalize.
        """
        _, solution, gold_answer = inp

        if not gold_answer or gold_answer.strip() == "":
            return 0.0

        if self.dataset_answer_format != "numeric":
            if self.dataset_answer_format == "boolean":
                pred_bool = _extract_boolean_answer(solution)
                gold_bool = _extract_boolean_answer(gold_answer)
                if pred_bool and gold_bool:
                    return 1.0 if pred_bool.lower() == gold_bool.lower() else 0.0
                return 0.0

            elif self.dataset_answer_format == "char":
                pred_char = _extract_single_letter_answer(solution)
                gold_char = _extract_single_letter_answer(gold_answer)
                if pred_char and gold_char:
                    return 1.0 if pred_char.upper() == gold_char.upper() else 0.0
                return 0.0

            elif self.dataset_answer_format == "string":
                pred_str = strip_string(solution) if solution else ""
                gold_str = strip_string(gold_answer) if gold_answer else ""
                return 1.0 if pred_str.lower() == gold_str.lower() else 0.0

            return 0.0

        try:
            if pre_extracted:
                pred = strip_string(
                    str(solution) if solution else "",
                    skip_unit=self.data_name in ["carp_en", "minerva_math"],
                )
            else:
                pred = extract_answer(solution, data_name=self.data_name)

            gold = _normalize_gold_answer(gold_answer, self.data_name)
            log.debug(
                f"_score_single BEFORE math_equal: pred={repr(pred)}, gold={repr(gold)}"
            )
            result = math_equal(pred, gold)
            log.debug(f"_score_single AFTER math_equal: result={result}")
            return 1.0 if result else 0.0

        except Exception as e:
            import traceback

            log.error(f"Math grading error: {e}\n{traceback.format_exc()}")
            return 0.0

    def __call__(self, problems: list[str], solutions: list[str], gold_answers: list[str]) -> list[float]:
        if not (len(problems) == len(solutions) == len(gold_answers)):
            raise ValueError("problems, solutions, and gold_answers must have the same length")

        inputs = list(zip(problems, solutions, gold_answers))

        if not self.silent:
            inputs = tqdm(inputs, total=len(inputs), desc="Verifying solutions")
        scores = [self._score_single(inp) for inp in inputs]
        if self.dataset_answer_format == "numeric":
            timeouts = get_timeout_count()
            if timeouts > 0:
                log.warning(
                    "Symbolic comparison timed out for %d samples during evaluation",
                    timeouts,
                )

        return scores