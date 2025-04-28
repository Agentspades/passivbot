import os
import numpy as np
import logging
import traceback
import asyncio
import math
from copy import deepcopy

import passivbot_rust as pbr
from optimize import (
    Evaluator,
    optimizer_overrides,
    individual_to_config,
    calc_hash,
    round_floats,
)
from backtest import (
    prep_backtest_args,
    expand_analysis,
)


class AsyncEvaluator(Evaluator):
    """Async version of the Evaluator class that properly awaits queue operations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure the results_queue is an asyncio.Queue
        if not isinstance(self.results_queue, asyncio.Queue):
            original_queue = self.results_queue
            self.results_queue = asyncio.Queue()
            logging.warning("Replaced non-asyncio queue with asyncio.Queue")

    async def evaluate(self, individual, overrides_list):
        """Async version of evaluate method that properly awaits queue operations"""
        if self.sig_digits > 0:
            individual[:] = [pbr.round_dynamic(v, self.sig_digits) for v in individual]
        config = individual_to_config(
            individual, optimizer_overrides, overrides_list, template=self.config
        )
        individual_hash = calc_hash(individual)
        if individual_hash in self.seen_hashes:
            existing_score = self.seen_hashes[individual_hash]
            self.duplicate_counter["count"] += 1
            dup_ct = self.duplicate_counter["count"]
            perturbation_funcs = [
                self.perturb_x_pct,
                self.perturb_step_digits,
                self.perturb_gaussian,
                self.perturb_random_subset,
                self.perturb_sample_some,
                self.perturb_large_uniform,
            ]
            for perturb_fn in perturbation_funcs:
                perturbed = round_floats(perturb_fn(individual), self.sig_digits)
                # changed = [(x, y) for x, y in zip(individual, perturbed) if x != y]
                # print("debug c", changed)
                new_hash = calc_hash(perturbed)
                if new_hash not in self.seen_hashes:
                    logging.info(
                        f"[DUPLICATE {dup_ct}] resolved with {perturb_fn.__name__} Hash: {new_hash}"
                    )
                    individual[:] = perturbed
                    self.seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed,
                        optimizer_overrides,
                        overrides_list,
                        template=self.config,
                    )
                    break
            else:
                logging.info(f"[DUPLICATE {dup_ct}] All perturbations failed.")
                if existing_score is not None:
                    return existing_score
        else:
            self.seen_hashes[individual_hash] = None
        analyses = {}
        for exchange in self.exchanges:
            bot_params, _, _ = prep_backtest_args(
                config,
                [],
                exchange,
                exchange_params=self.exchange_params[exchange],
                backtest_params=self.backtest_params[exchange],
            )
            fills, equities_usd, equities_btc, analysis_usd, analysis_btc = (
                pbr.run_backtest(
                    self.shared_memory_files[exchange],
                    self.hlcvs_shapes[exchange],
                    self.hlcvs_dtypes[exchange].str,
                    self.btc_usd_shared_memory_files[exchange],
                    self.btc_usd_dtypes[exchange].str,
                    bot_params,
                    self.exchange_params[exchange],
                    self.backtest_params[exchange],
                )
            )
            analyses[exchange] = expand_analysis(
                analysis_usd, analysis_btc, fills, config
            )
        analyses_combined = self.combine_analyses(analyses)
        objectives = self.calc_fitness(analyses_combined)
        for i, val in enumerate(objectives):
            analyses_combined[f"w_{i}"] = val
        data = {
            **config,
            "analyses_combined": analyses_combined,
            "analyses": analyses,
        }
        # Use await with the queue put operation
        await self.results_queue.put(data)
        actual_hash = calc_hash(individual)
        self.seen_hashes[actual_hash] = tuple(objectives)
        return tuple(objectives)

    # Include all the perturbation methods from the original Evaluator class
    # These methods don't need to be async since they don't use the queue

    def perturb_step_digits(self, individual, change_chance=0.5):
        perturbed = []
        for i, val in enumerate(individual):
            if np.random.random() < change_chance:  # x% chance of leaving unchanged
                perturbed.append(val)
                continue
            low, high = self.param_bounds_expanded[i]
            if high == low:
                perturbed.append(val)
                continue

            if val != 0.0:
                exponent = math.floor(math.log10(abs(val))) - (self.sig_digits - 1)
                step = 10**exponent
            else:
                step = (high - low) * 10 ** -(self.sig_digits - 1)

            direction = np.random.choice([-1.0, 1.0])
            new_val = pbr.round_dynamic(val + step * direction, self.sig_digits)
            new_val = min(max(new_val, low), high)
            perturbed.append(new_val)

        return perturbed

    def perturb_x_pct(self, individual, magnitude=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            low, high = self.param_bounds_expanded[i]
            if high == low:
                perturbed.append(val)
                continue
            new_val = val * (1 + np.random.uniform(-magnitude, magnitude))
            new_val = min(max(pbr.round_dynamic(new_val, self.sig_digits), low), high)
            perturbed.append(new_val)
        return perturbed

    def perturb_random_subset(self, individual, frac=0.2):
        perturbed = individual.copy()
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            low, high = self.param_bounds_expanded[i]
            if low != high:
                delta = (high - low) * 0.01
                step = delta * np.random.uniform(-1.0, 1.0)
                val = individual[i] + step
                perturbed[i] = pbr.round_dynamic(
                    np.clip(val, low, high), self.sig_digits
                )
        return perturbed

    def perturb_sample_some(self, individual, frac=0.2):
        perturbed = individual.copy()
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            low, high = self.param_bounds_expanded[i]
            if low != high:
                perturbed[i] = pbr.round_dynamic(
                    np.random.uniform(low, high), self.sig_digits
                )
        return perturbed

    def perturb_gaussian(self, individual, scale=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            low, high = self.param_bounds_expanded[i]
            if high == low:
                perturbed.append(val)
                continue
            noise = np.random.normal(0, scale * (high - low))
            new_val = pbr.round_dynamic(val + noise, self.sig_digits)
            new_val = min(max(new_val, low), high)
            perturbed.append(new_val)
        return perturbed

    def perturb_large_uniform(self, individual):
        perturbed = []
        for i in range(len(individual)):
            low, high = self.param_bounds_expanded[i]
            if low == high:
                perturbed.append(low)
            else:
                perturbed.append(
                    pbr.round_dynamic(np.random.uniform(low, high), self.sig_digits)
                )
        return perturbed
