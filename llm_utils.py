# llm_utils.py
import os
import json
import re
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMIndividualGenerator:
    """
    Generate GP expressions from the current best individual using an LLM API.
    The output expression must be compatible with DEAP gp.PrimitiveTree.from_string.
    """

    def __init__(
        self,
        model="deepseek-chat",
        api_base=None,
        api_key=None,
        temperature=0.7,
        max_retries=3,
        primitive_names=None,
        n_features=None,
    ):
        if OpenAI is None:
            raise ImportError(
                "Please install openai first: pip install openai"
            )

        self.model = model
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self.primitive_names = primitive_names or {"add", "subtract", "multiply", "sqrt", "inv"}
        self.n_features = n_features

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)

    # ---------------------------------------------------------------------
    # Explore prompts
    # ---------------------------------------------------------------------
    def _build_system_prompt_explore(self):
        return f"""
You are helping generate exploratory symbolic regression expressions for DEAP genetic programming.

You must output ONLY a single valid expression string, with no explanation.

Allowed function names:
{sorted(list(self.primitive_names))}

Allowed terminals:
- ARG0, ARG1, ..., ARG{self.n_features - 1}
- numeric constants like 1.0, -0.5, 2

Expression format examples:
- add(ARG0, ARG1)
- multiply(add(ARG0, 1.0), sqrt(ARG2))
- inv(add(ARG1, -1.0))
- subtract(multiply(ARG0, ARG1), sqrt(ARG2))

Rules:
1. Output exactly one expression.
2. Do not use any function outside the allowed set.
3. Do not use variable names other than ARG0...ARG{self.n_features - 1}.
4. Keep syntax valid for DEAP gp.PrimitiveTree.from_string.
5. Generate a structurally DIFFERENT expression from the current best.
6. Try different variable combinations, nesting patterns, or high-level composition.
7. Keep the expression reasonably compact and plausible.
8. Avoid copying the provided best expression with only trivial edits.
""".strip()

    def _build_user_prompt_explore(self, best_expression, task_index, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        feature_stats_text = []
        max_show = min(10, X.shape[1])
        for i in range(max_show):
            feature_stats_text.append(
                f"ARG{i}: mean={feature_means[i]:.4f}, std={feature_stds[i]:.4f}"
            )
        feature_stats_text = "\n".join(feature_stats_text)

        return f"""
Task index: {task_index}

Mode: EXPLORE

Goal:
Escape local optimum by proposing a new symbolic expression that is meaningfully different from the current best.

Current best expression:
{best_expression}

Target summary:
y mean = {y_mean:.6f}
y std = {y_std:.6f}
y min = {y_min:.6f}
y max = {y_max:.6f}

Feature summary:
{feature_stats_text}

Requirements:
- Make the new expression structurally different from the current best.
- You may change variable usage, nesting pattern, and high-level decomposition.
- Keep it valid, compact, and plausible.

Return only one valid expression string.
""".strip()

    # ---------------------------------------------------------------------
    # Refine prompts
    # ---------------------------------------------------------------------
    def _build_system_prompt_refine(self):
        return f"""
You are helping refine symbolic regression expressions for DEAP genetic programming.

You must output ONLY a single valid expression string, with no explanation.

Allowed function names:
{sorted(list(self.primitive_names))}

Allowed terminals:
- ARG0, ARG1, ..., ARG{self.n_features - 1}
- numeric constants like 1.0, -0.5, 2

Expression format examples:
- add(ARG0, ARG1)
- multiply(add(ARG0, 1.0), sqrt(ARG2))
- inv(add(ARG1, -1.0))
- subtract(multiply(ARG0, ARG1), sqrt(ARG2))

Rules:
1. Output exactly one expression.
2. Do not use any function outside the allowed set.
3. Do not use variable names other than ARG0...ARG{self.n_features - 1}.
4. Keep syntax valid for DEAP gp.PrimitiveTree.from_string.
5. Keep the new expression structurally CLOSE to the provided best expression.
6. Prefer small, local edits such as subtree replacement, small simplification, light transformation, or constant adjustment.
7. Avoid drastic structural rewrites.
8. Keep the expression reasonably compact.
""".strip()

    def _build_user_prompt_refine(self, best_expression, task_index, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        feature_stats_text = []
        max_show = min(10, X.shape[1])
        for i in range(max_show):
            feature_stats_text.append(
                f"ARG{i}: mean={feature_means[i]:.4f}, std={feature_stds[i]:.4f}"
            )
        feature_stats_text = "\n".join(feature_stats_text)

        return f"""
Task index: {task_index}

Mode: REFINE

Goal:
Improve the current best expression with a small, local modification.

Current best expression:
{best_expression}

Target summary:
y mean = {y_mean:.6f}
y std = {y_std:.6f}
y min = {y_min:.6f}
y max = {y_max:.6f}

Feature summary:
{feature_stats_text}

Requirements:
- Stay structurally close to the current best.
- Prefer a small edit rather than a full redesign.
- Possible edits include replacing one subtree, inserting/removing a light transformation, simplifying a part, or adjusting a constant.

Return only one valid expression string.
""".strip()

    def _extract_expression(self, text):
        text = text.strip()

        text = re.sub(r"^```[a-zA-Z]*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "expression" in obj:
                text = obj["expression"].strip()
        except Exception:
            pass

        return text

    def generate_from_best(self, best_expression, mode, task_index, X, y):
        if mode == "explore":
            system_prompt = self._build_system_prompt_explore()
            user_prompt = self._build_user_prompt_explore(
                best_expression=best_expression,
                task_index=task_index,
                X=X,
                y=y,
            )
        else:
            system_prompt = self._build_system_prompt_refine()
            user_prompt = self._build_user_prompt_refine(
                best_expression=best_expression,
                task_index=task_index,
                X=X,
                y=y,
            )

        last_err = None
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.choices[0].message.content
                expr = self._extract_expression(text)
                return expr
            except Exception as e:
                last_err = e

        raise RuntimeError(f"LLM generation failed after retries: {last_err}")