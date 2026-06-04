import numpy as np
import math
import copy
import random
import operator
from deap import base, creator, gp, tools
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.spatial.distance import cityblock
from gp2 import SSC, staticLimit_multi, SSC_mate, quick_evaluate_subtree
from functools import partial
import re
from llm_utils import LLMIndividualGenerator


# test upload by github
class GPIndividualEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, pset, individual):
        self.pset = pset
        self.individual = individual
        self.func = gp.compile(self.individual, self.pset)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([self.func(*x) for x in X])


class SymbolicRegressorGP(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible class for symbolic regression using DEAP."""

    def __init__(
            self,
            n_generations=100,
            pop_size=100,
            crossover_prob=0.9,
            mutation_prob=0.1,
            n_selected_features=None,
            verbose=False,
            tr=0.3,
            semantic_crossover=True,
            use_llm=False,
            llm_start_gen=None,
            llm_ratio=0.1,
            llm_model="deepseek-chat",
            llm_api_base=None,
            llm_api_key=None,
            llm_temperature=0.7,
            llm_max_retries=3,
            llm_debug=False,
            llm_log_file=None,
            llm_retry_on_fail=1,
            llm_explore_min_semantic_dist=0.05,
            llm_stagnation_window=5,
            llm_stagnation_epsilon=1e-4,
            llm_low_diversity_epsilon=1e-4,
            llm_refine_top_k=10,
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_selected_features = n_selected_features
        self.verbose = verbose
        self.toolbox = base.Toolbox()
        self.tr = tr
        self.semantic_crossover = semantic_crossover

        # LLM options
        self.use_llm = use_llm
        self.llm_start_gen = llm_start_gen if llm_start_gen is not None else n_generations // 2
        self.llm_ratio = llm_ratio
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_temperature = llm_temperature
        self.llm_max_retries = llm_max_retries
        self.llm_debug = llm_debug
        self.llm_log_file = llm_log_file
        self.llm_generator = None

        # runtime states
        self.current_gen = 0
        self._llm_activated_once = False
        self.llm_retry_on_fail = llm_retry_on_fail
        self.llm_explore_min_semantic_dist = llm_explore_min_semantic_dist
        self.llm_stagnation_window = llm_stagnation_window
        self.llm_stagnation_epsilon = llm_stagnation_epsilon
        self.llm_low_diversity_epsilon = llm_low_diversity_epsilon
        self.llm_refine_top_k = llm_refine_top_k

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------
    def _log_llm(self, msg, force=False):
        if force or self.llm_debug:
            print(msg)
        if self.llm_log_file is not None:
            with open(self.llm_log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _fitness_to_float(self, fitness):
        if hasattr(fitness, "values"):
            vals = fitness.values
            if isinstance(vals, tuple) and len(vals) > 0:
                return float(vals[0])
        if isinstance(fitness, (tuple, list)):
            return float(fitness[0])
        return float(fitness)

    def _describe_individual(self, ind):
        try:
            return f"{str(ind)} | size={len(ind)} | height={ind.height}"
        except Exception:
            return str(ind)

    # -------------------------------------------------------------------------
    # Original methods
    # -------------------------------------------------------------------------
    def SSD(self, parent_a, parent_b):
        func_a = gp.compile(parent_a, self.pset)
        func_b = gp.compile(parent_b, self.pset)
        y_pred_a = []
        y_pred_b = []
        for row in self.X:
            try:
                y_pred_a.append(func_a(*row))
            except:
                y_pred_a.append(np.nan)

        for row in self.X:
            try:
                y_pred_b.append(func_b(*row))
            except:
                y_pred_b.append(np.nan)

        Distance = cityblock(y_pred_a, y_pred_b) / len(self.X)
        return Distance

    # -------------------------------------------------------------------------
    # LLM helper methods
    # -------------------------------------------------------------------------
    def _get_allowed_primitive_names(self):
        return {"add", "subtract", "multiply", "sqrt", "inv"}

    def _sanitize_llm_expression(self, expr: str) -> str:
        expr = expr.strip()
        expr = re.sub(r"```(?:python)?", "", expr)
        expr = expr.replace("```", "")
        expr = expr.replace("np.", "")
        expr = expr.replace("numpy.", "")
        expr = expr.replace("^", "**")
        expr = re.sub(r"^\s*(Expression|Candidate|Best|Output)\s*:\s*", "", expr, flags=re.I)
        lines = [line.strip() for line in expr.splitlines() if line.strip()]
        if len(lines) > 0:
            expr = lines[0]
        return expr.strip()

    def _expr_to_individual(self, expr: str):
        expr = self._sanitize_llm_expression(expr)
        tree = gp.PrimitiveTree.from_string(expr, self.pset)
        ind = creator.Individual(tree)
        return ind

    def _is_finite_fitness(self, fitness_value):
        return np.isfinite(fitness_value)

    def _predict_individual_on_X(self, ind):
        func = gp.compile(ind, self.pset)
        preds = []
        for row in self.X:
            try:
                val = func(*row)
            except:
                val = np.nan
            preds.append(val)
        return np.asarray(preds, dtype=np.float64)

    def _semantic_distance_between_individuals(self, ind_a, ind_b):
        pred_a = self._predict_individual_on_X(ind_a)
        pred_b = self._predict_individual_on_X(ind_b)

        pred_a = np.nan_to_num(pred_a, nan=0.0, posinf=1e6, neginf=-1e6)
        pred_b = np.nan_to_num(pred_b, nan=0.0, posinf=1e6, neginf=-1e6)

        return cityblock(pred_a, pred_b) / max(1, len(pred_a))

    def _current_task_mean_fitness(self, task):
        pop = self.sub_population[task]
        fits = [self._fitness_to_float(ind.fitness) for ind in pop]
        return float(np.mean(fits))

    def _current_task_fitness_std(self, task):
        pop = self.sub_population[task]
        fits = [self._fitness_to_float(ind.fitness) for ind in pop]
        return float(np.std(fits))

    def _get_task_best_history(self, task):
        hist = []
        if task >= len(self.Best_of_individual):
            return hist
        for ind in self.Best_of_individual[task]:
            if ind is not None and hasattr(ind, "fitness") and ind.fitness.valid:
                hist.append(self._fitness_to_float(ind.fitness))
        return hist

    def _passes_explore_semantic_filter(self, candidate, task):
        best_ind = tools.selBest(self.sub_population[task], k=1)[0]
        dist = self._semantic_distance_between_individuals(candidate, best_ind)

        y_range = float(self.y[:, task].max() - self.y[:, task].min())
        if y_range <= 0:
            y_range = 1.0

        threshold = self.llm_explore_min_semantic_dist * y_range
        passed = dist >= threshold

        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Explore semantic distance="
            f"{dist:.6f}, threshold={threshold:.6f}, passed={passed}",
            force=True
        )
        return passed

    def _get_task_topk_threshold(self, task, k=None):
        if k is None:
            k = self.llm_refine_top_k

        pop = self.sub_population[task]
        if len(pop) == 0:
            return np.inf

        k = max(1, min(k, len(pop)))
        topk = tools.selBest(pop, k=k)
        threshold = max(self._fitness_to_float(ind.fitness) for ind in topk)

        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Refine top-k threshold "
            f"(k={k}) = {threshold:.6f}",
            force=True
        )
        return threshold

    def _accept_llm_candidate(self, ind, task, mode):
        cand_fit = self._fitness_to_float(ind.fitness)

        if not self._is_finite_fitness(cand_fit):
            self._log_llm(
                f"[Gen {self.current_gen}] [Task {task}] Candidate rejected: non-finite fitness {cand_fit}",
                force=True
            )
            return False, f"non-finite fitness: {cand_fit}"

        if mode == "explore":
            if not self._passes_explore_semantic_filter(ind, task):
                return False, "explore semantic distance too small"

            pop = self.sub_population[task]
            pop_fits = []
            for p in pop:
                f = self._fitness_to_float(p.fitness)
                if np.isfinite(f):
                    pop_fits.append(f)

            if len(pop_fits) == 0:
                return False, "no valid population fitness for explore acceptance"

            mean_fit = float(np.mean(pop_fits))

            if cand_fit >= mean_fit:
                return False, f"explore fitness not better than mean: cand={cand_fit:.6f}, mean={mean_fit:.6f}"

            self._log_llm(
                f"[Gen {self.current_gen}] [Task {task}] Explore candidate accepted. "
                f"candidate_fit={cand_fit:.6f}, mean_fit={mean_fit:.6f}",
                force=True
            )
            return True, "accepted by explore rule (semantic + better than mean)"

        elif mode == "refine":
            topk_threshold = self._get_task_topk_threshold(task, self.llm_refine_top_k)
            if cand_fit < topk_threshold:
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Refine candidate accepted: "
                    f"candidate_fit={cand_fit:.6f} < topk_threshold={topk_threshold:.6f}",
                    force=True
                )
                return True, "accepted by refine top-k rule"
            else:
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Refine candidate rejected: "
                    f"candidate_fit={cand_fit:.6f} >= topk_threshold={topk_threshold:.6f}",
                    force=True
                )
                return False, (
                    f"candidate fitness {cand_fit:.6f} is not better than "
                    f"top-{self.llm_refine_top_k} threshold {topk_threshold:.6f}"
                )

        else:
            return False, f"unknown mode: {mode}"

    def _pick_llm_mode(self, task, gen):
        history = self._get_task_best_history(task)
        fitness_std = self._current_task_fitness_std(task)

        stagnated = False
        if len(history) >= self.llm_stagnation_window + 1:
            old_best = history[-self.llm_stagnation_window - 1]
            recent_best = history[-1]
            improvement = old_best - recent_best
            stagnated = improvement < self.llm_stagnation_epsilon
        else:
            improvement = None

        low_diversity = fitness_std < self.llm_low_diversity_epsilon

        self._log_llm(
            f"[Gen {gen}] [Task {task}] Mode decision state | "
            f"history_len={len(history)} | improvement={improvement} | "
            f"fitness_std={fitness_std:.6f} | stagnated={stagnated} | low_diversity={low_diversity}",
            force=True
        )

        if stagnated or low_diversity:
            return "explore" if random.random() < 0.7 else "refine"
        else:
            return "refine" if random.random() < 0.7 else "explore"

    def _generate_llm_individual(self, task, mode="explore"):
        best_ind = tools.selBest(self.sub_population[task], k=1)[0]
        best_expr = str(best_ind)
        mean_fit = self._current_task_mean_fitness(task)

        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] LLM mode={mode}",
            force=True
        )
        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Current best expression: {best_expr}",
            force=True
        )
        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Current best fitness: {self._fitness_to_float(best_ind.fitness):.6f}",
            force=True
        )
        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Current population mean fitness: {mean_fit:.6f}",
            force=True
        )

        max_attempts = 1 + self.llm_retry_on_fail
        last_reason = None

        for attempt in range(1, max_attempts + 1):
            try:
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] LLM attempt {attempt}/{max_attempts}",
                    force=True
                )

                new_expr = self.llm_generator.generate_from_best(
                    best_expression=best_expr,
                    mode=mode,
                    task_index=task,
                    X=self.X,
                    y=self.y[:, task],
                )

                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] LLM raw expression: {new_expr}",
                    force=True
                )

                sanitized_expr = self._sanitize_llm_expression(new_expr)
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Sanitized expression: {sanitized_expr}",
                    force=True
                )

                ind = self._expr_to_individual(sanitized_expr)
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Parsed LLM individual: {self._describe_individual(ind)}",
                    force=True
                )

                del ind.fitness.values
                ind.fitness.values = self.evaluate_individual(ind, task)
                cand_fit = self._fitness_to_float(ind.fitness)

                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] LLM individual fitness: {cand_fit:.6f}",
                    force=True
                )

                accepted, reason = self._accept_llm_candidate(ind, task, mode)
                if accepted:
                    self._log_llm(
                        f"[Gen {self.current_gen}] [Task {task}] LLM individual accepted for injection pool.",
                        force=True
                    )
                    return ind
                else:
                    last_reason = reason
                    self._log_llm(
                        f"[Gen {self.current_gen}] [Task {task}] Attempt failed: {last_reason}",
                        force=True
                    )
                    continue

            except Exception as e:
                last_reason = repr(e)
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Attempt failed with error={last_reason}",
                    force=True
                )

        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] All LLM attempts failed. Fallback triggered. "
            f"Last reason: {last_reason}",
            force=True
        )

        ind = self.toolbox.individual()
        ind.fitness.values = self.evaluate_individual(ind, task)

        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Fallback random individual: {self._describe_individual(ind)}",
            force=True
        )
        self._log_llm(
            f"[Gen {self.current_gen}] [Task {task}] Fallback fitness: {self._fitness_to_float(ind.fitness):.6f}",
            force=True
        )

        return ind

    def _generate_llm_offspring_for_task(self, task, gen, n_llm):
        llm_offspring = []
        if n_llm <= 0:
            return llm_offspring

        self._log_llm(
            f"[Gen {gen}] [Task {task}] LLM will generate {n_llm} individual(s).",
            force=True
        )

        for idx in range(n_llm):
            mode = self._pick_llm_mode(task, gen)
            self._log_llm(
                f"[Gen {gen}] [Task {task}] LLM request #{idx + 1}/{n_llm}, mode={mode}",
                force=True
            )
            ind = self._generate_llm_individual(task, mode=mode)
            setattr(ind, "_llm_mode", mode)
            llm_offspring.append(ind)

        return llm_offspring

    def _merge_llm_offspring_into_task_population(self, task, llm_offspring):
        if len(llm_offspring) == 0:
            return

        pop = self.sub_population[task]

        for ind in llm_offspring:
            mode = getattr(ind, "_llm_mode", "explore")
            cand_fit = self._fitness_to_float(ind.fitness)

            if not self._is_finite_fitness(cand_fit):
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Rejected due to non-finite candidate fitness: {cand_fit}",
                    force=True
                )
                continue

            accepted, reason = self._accept_llm_candidate(ind, task, mode)
            if not accepted:
                self._log_llm(
                    f"[Gen {self.current_gen}] [Task {task}] Candidate rejected during merge: {reason}",
                    force=True
                )
                continue

            worst_idx = max(range(len(pop)), key=lambda i: self._fitness_to_float(pop[i].fitness))
            worst_fit = self._fitness_to_float(pop[worst_idx].fitness)
            worst_expr = str(pop[worst_idx])

            self._log_llm(
                f"[Gen {self.current_gen}] [Task {task}] LLM individual merged into population "
                f"(mode={mode}, candidate_fit={cand_fit:.6f}).",
                force=True
            )
            self._log_llm(
                f"[Gen {self.current_gen}] [Task {task}] Replacing worst individual: "
                f"{worst_expr} | worst_fit={worst_fit:.6f}",
                force=True
            )
            pop[worst_idx] = ind

        self.sub_population[task] = pop

    # -------------------------------------------------------------------------
    # Main EA
    # -------------------------------------------------------------------------
    def eaSimple_Elitism(
            self,
            population,
            toolbox,
            cxpb,
            mutpb,
            ngen,
            stats=None,
            halloffame=None,
            verbose=__debug__,
    ):
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        self.sub_population = [0 for k in range(self.y.shape[1])]
        offspring = [0 for k in range(self.y.shape[1])]
        self.Best_of_individual = [0 for k in range(self.y.shape[1])]

        # split a population into k subpopulations
        for k in range(self.y.shape[1]):
            self.sub_population[k] = []
            offspring[k] = []
            self.Best_of_individual[k] = []

        # Evaluate the individuals with an invalid fitness
        j = 0
        for ind in zip(invalid_ind):
            t = math.floor(j / math.floor(self.pop_size))
            if t == self.y.shape[1]:
                t = self.y.shape[1] - 1
            ind[0].fitness.values = self.evaluate_individual(ind[0], t)
            self.sub_population[t].append(ind[0])
            j = j + 1

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for i in range(self.y.shape[1]):
            top = tools.selBest(self.sub_population[i], k=1)

        # Begin the generational process
        self.redunt_time = 0
        for gen in range(1, ngen + 1):
            self.current_gen = gen

            if self.use_llm and gen >= self.llm_start_gen and not self._llm_activated_once:
                self._log_llm(f"[Gen {gen}] LLM activated.", force=True)
                self._llm_activated_once = True

            new_population = []
            len_invalid = 0

            for task in range(self.y.shape[1]):
                len_sub_population = self.pop_size
                offspring_selected = toolbox.select(
                    self.sub_population[task], len_sub_population
                )
                clone_offspring = copy.deepcopy(offspring_selected)
                clone_offspring_mut = copy.deepcopy(offspring_selected)

                offspring[task] = []

                # crossover
                for j in range(int(len_sub_population)):
                    if len(offspring[task]) >= 100:
                        break

                    if random.random() < cxpb:
                        # crossover in different tasks
                        if random.random() <= self.tr:
                            parent_a = toolbox.select(self.sub_population[task], 1)
                            parent_a_copy = copy.deepcopy(parent_a)

                            while True:
                                random_t = random.randint(0, self.y.shape[1] - 1)
                                if random_t != task:
                                    break

                            parent_b = toolbox.select(self.sub_population[random_t], 1)
                            parent_b_copy = copy.deepcopy(parent_b)

                            # semantic similarity in crossover
                            child_a, child_b = toolbox.semantic(
                                parent_a_copy[0], parent_b_copy[0], self.y[:, task]
                            )

                            del child_a.fitness.values, child_b.fitness.values
                            offspring[task].append(child_a)
                            # offspring[task].append(child_b)

                        # crossover in the same task
                        else:
                            parent_a = copy.deepcopy(clone_offspring[j])
                            try:
                                parent_b = copy.deepcopy(clone_offspring[j + 1])
                                j = j + 1
                            except:
                                continue

                            child_a, child_b = toolbox.mate(parent_a, parent_b)
                            del child_a.fitness.values, child_b.fitness.values
                            offspring[task].append(child_a)
                            offspring[task].append(child_b)

                    else:
                        parent = toolbox.select(clone_offspring_mut, 1)
                        selected_parent = copy.deepcopy(parent)
                        if random.random() < mutpb:
                            (selected_parent,) = toolbox.mutate(selected_parent[0])
                            del selected_parent.fitness.values
                            offspring[task].append(selected_parent)

                invalid_ind = [ind for ind in offspring[task] if not ind.fitness.valid]

                seen_individuals = set()
                new_unique_ind = []
                for ind in zip(invalid_ind):
                    ind_str = str(ind[0])
                    if ind_str not in seen_individuals:
                        seen_individuals.add(ind_str)
                        ind[0].fitness.values = self.evaluate_individual(ind[0], task)
                        new_unique_ind.append(ind[0])
                        len_invalid = len_invalid + 1

                offspring[task] = new_unique_ind

                # Elitism
                elitism = tools.selBest(self.sub_population[task], k=1)
                for i in range(len(elitism)):
                    offspring[task].append(elitism[i])

                if int(len_sub_population - len(offspring[task])) > 0:
                    rest = self.toolbox.population(n=int(len_sub_population - len(offspring[task])))
                    for i in range(int(len_sub_population - len(offspring[task]))):
                        rest[i].fitness.values = self.evaluate_individual(rest[i], task)
                        len_invalid = len_invalid + 1
                        offspring[task].append(rest[i])

                self.sub_population[task] = offspring[task]

                # ---------------- LLM injection starts here ----------------
                if self.use_llm and gen >= self.llm_start_gen:
                    n_llm = max(1, int(self.llm_ratio * self.pop_size))
                    llm_offspring = self._generate_llm_offspring_for_task(task, gen, n_llm)
                    self._merge_llm_offspring_into_task_population(task, llm_offspring)
                # ---------------- LLM injection ends here ----------------

                for i in range(len(self.sub_population[task])):
                    new_population.append(self.sub_population[task][i])

            # Append the current generation statistics to the logbook
            record = stats.compile(new_population) if stats else {}
            logbook.record(gen=gen, nevals=len_invalid, **record)
            if verbose:
                print(logbook.stream)

            # Store best-of-individual in every generation
            for i in range(self.y.shape[1]):
                top = tools.selBest(self.sub_population[i], k=1)
                self.Best_of_individual[i].append(top[0])

                if self.use_llm and gen >= self.llm_start_gen:
                    self._log_llm(
                        f"[Gen {gen}] [Task {i}] Post-LLM best: {str(top[0])}",
                        force=True
                    )
                    self._log_llm(
                        f"[Gen {gen}] [Task {i}] Post-LLM best fitness: {self._fitness_to_float(top[0].fitness):.6f}",
                        force=True
                    )

        return new_population, logbook

    # -------------------------------------------------------------------------
    # sklearn fit
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X = X
        self.y = y

        # reset runtime states
        self.current_gen = 0
        self._llm_activated_once = False

        # Define primitive set
        pset = gp.PrimitiveSet("MAIN", X.shape[1])
        self.add_functions_to_pset(pset)
        pset.addEphemeralConstant("rand101", lambda: np.random.randint(-1, 1))
        self.pset = pset

        # Define individual and population
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        class PrimitiveTreeCopy(gp.PrimitiveTree):
            def __deepcopy__(self, memo):
                new_content = copy.deepcopy(self[:], memo)
                new_instance = self.__class__(new_content)
                copied_dict = {
                    key: (
                        value
                        if key in ("semantics", "subtree_semantics")
                        else copy.deepcopy(value, memo)
                    )
                    for key, value in self.__dict__.items()
                }
                new_instance.__dict__.update(copied_dict)
                return new_instance

        if not hasattr(creator, "Individual"):
            creator.create("Individual", PrimitiveTreeCopy, fitness=creator.FitnessMin)

        self.define_toolbox()

        if self.use_llm:
            self.llm_generator = LLMIndividualGenerator(
                model=self.llm_model,
                api_base=self.llm_api_base,
                api_key=self.llm_api_key,
                temperature=self.llm_temperature,
                max_retries=self.llm_max_retries,
                primitive_names=self._get_allowed_primitive_names(),
                n_features=X.shape[1],
            )
            self._log_llm(
                f"[Init] LLM enabled | model={self.llm_model} | start_gen={self.llm_start_gen} | ratio={self.llm_ratio} | refine_top_k={self.llm_refine_top_k}",
                force=True
            )
        else:
            self.llm_generator = None
            self._log_llm("[Init] LLM disabled.", force=True)

        # Initialize population: 保持原文逻辑
        pop = self.toolbox.population(n=self.pop_size * self.y.shape[1])

        # Define the statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(lambda ind: ind.height)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        hof = tools.HallOfFame(1)

        self.eaSimple_Elitism(
            pop,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=mstats,
            halloffame=hof,
            verbose=self.verbose,
        )

        # Obtain the best model based on standard GP
        self.top = [0 for k in range(self.y.shape[1])]
        self.final_model = [0 for k in range(self.y.shape[1])]
        for i in range(self.y.shape[1]):
            self.top[i] = tools.selBest(self.sub_population[i], k=1)
            self.final_model[i] = self.top[i][0]

            if self.use_llm:
                self._log_llm(
                    f"[Final] [Task {i}] Best individual: {str(self.final_model[i])}",
                    force=True
                )
                self._log_llm(
                    f"[Final] [Task {i}] Best fitness: {self._fitness_to_float(self.final_model[i].fitness):.6f}",
                    force=True
                )

        return self

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    def evaluate_individual(self, individual, t):
        y_pred, y_subtree_semantics = quick_evaluate_subtree(
            individual, self.pset, self.X
        )
        individual.subtree_semantics = {k: v for k, v in y_subtree_semantics}

        return (
            math.sqrt(np.mean((self.y[:, t] - y_pred) ** 2))
            / (self.y[:, t].max() - self.y[:, t].min()),
        )

    # -------------------------------------------------------------------------
    # Primitive set
    # -------------------------------------------------------------------------
    def add_functions_to_pset(self, pset):
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        # pset.addPrimitive(self.protectedDiv, 2)
        pset.addPrimitive(self.sqrt, 1)
        pset.addPrimitive(self.inv, 1)

    def inv(self, x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x) > 0, np.divide(1, x), 1)

    def protectedDiv(self, x1, x2):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x2) > 0, np.divide(x1, x2), 1)

    def sqrt(self, x):
        return np.sqrt(np.abs(x))

    # -------------------------------------------------------------------------
    # Toolbox
    # -------------------------------------------------------------------------
    def define_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=6)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("semantic", partial(SSC, pset=self.pset, X=self.X))
        if self.semantic_crossover:
            self.toolbox.register("mate", partial(SSC_mate, pset=self.pset, X=self.X))
        else:
            self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=8)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.register("select", tools.selTournament, tournsize=7)

        # 保持原文逻辑：只限制 mutate，mate / semantic 不恢复
        # self.toolbox.decorate(
        #     "mate", staticLimit_multi(key=operator.attrgetter("height"), max_value=10)
        # )
        # self.toolbox.decorate(
        #     "semantic",
        #     staticLimit_multi(key=operator.attrgetter("height"), max_value=10),
        # )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8)
        )

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------
    def predict_Standard(self, X, func):
        y_pred = []
        for row in X:
            try:
                y_pred.append(func(*row))
            except:
                y_pred.append(np.nan)
        return y_pred