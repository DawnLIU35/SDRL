# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import time
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from ray.exceptions import RayActorError
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, process_validation_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.torch_functional import pad_sequence_to_length

SECOND_TURN_USER_PROMPT_TEMPLATE = (
    "Now another agent provided a different solution.\n\n"
    "[AGENT RESPONSE START]\n"
    "{partner_response}\n"
    "[AGENT RESPONSE END]\n\n"
    "You are continuing the debate from your previous answer. Your goals are:\n"
    "1) Identify and list the key points where the agent's reasoning or final answer conflicts with yours.\n"
    "2) Critique the conflicts step by step: point out specific incorrect steps, hidden assumptions, "
    "or arithmetic/algebra errors involved in those conflicts, and explain precisely why they are wrong.\n"
    "3) If the agent's response is correct and your previous answer was wrong, explicitly acknowledge that and switch "
    "to the correct answer. Otherwise, defend your previous answer.\n"
    "4) Provide a clean, self-contained step-by-step solution that resolves the disagreement "
    "(you may reuse correct parts from either response, but don't skip logic).\n"
    "5) The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _compute_advantage_with_turns(self, batch: DataProto, norm_adv_by_std_in_grpo: bool) -> DataProto:
        """
        Compute advantages separately for first- and second-turn responses.
        Falls back to the standard computation when only first-turn responses are present.
        """
        turn_ids = batch.non_tensor_batch.get("__num_turns__")
        turn_ids_np = np.asarray(turn_ids) if turn_ids is not None else None

        if turn_ids_np is None or not np.any(turn_ids_np != 1) or self.config.self_debate.enable_debate_training is False:
            return compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        adv_estimator = self.config.algorithm.adv_estimator
        gamma = self.config.algorithm.gamma
        lam = self.config.algorithm.lam
        num_repeat = self.config.actor_rollout_ref.rollout.n

        first_mask_np = turn_ids_np == 1
        second_mask_np = ~first_mask_np

        computed_batches: list[DataProto] = []

        if np.any(first_mask_np):
            first_batch = batch[first_mask_np]
            first_batch = compute_advantage(
                first_batch,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            computed_batches.append(first_batch)

        if np.any(second_mask_np):
            second_batch = batch[second_mask_np]
            adv_fn = getattr(self.config.self_debate, "second_turn_advantage", "grpo")
            adv_fn = str(adv_fn).lower()
            if adv_fn != "grpo":
                raise ValueError(f"Unsupported second_turn_advantage: {adv_fn}")
            second_batch = compute_advantage(
                second_batch,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

            computed_batches.append(second_batch)

        return computed_batches[0] if len(computed_batches) == 1 else DataProto.concat(computed_batches)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX: # False
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    if (
                        self.config.self_debate.enable_debate_training
                        and "__num_turns__" not in batch.non_tensor_batch
                    ):
                        batch.non_tensor_batch["__num_turns__"] = np.ones(len(batch), dtype=np.int32)

                    # Build and append second-turn debate trajectories
                    if self.config.self_debate.enable_debate_training:
                        with marked_timer("debate_train", timing_raw, "purple"):
                            batch = self._generate_training_second_turn_batch(batch, metrics, timing_raw)
                        if batch is None:
                            continue

                    # TODO ours: replace the one-turn with two trun debate batch
                    size_divisor = (
                        self.actor_rollout_wg.world_size
                        if not self.async_rollout_mode
                        else self.config.actor_rollout_ref.rollout.agent.num_workers
                    )
                    batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout IS weights and mismatch metrics (inherited from RayPPOTrainer)
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    # IS and mismatch metrics already have mismatch/ prefix
                    metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = self._compute_advantage_with_turns(
                            batch=batch, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    def _construct_second_turn_inputs(
        self,
        base_batch: DataProto,
        include_extra_metadata: bool,
        tokenizer_kwargs: dict | None = None,
        meta_info: dict | None = None,
        reward_result: dict | None = None,
        max_initial_response_length: int | None = None,
    ):
        """
        train_batch.batch: attention_mask, input_ids, position_ids, responses, prompts, token_level_scores, token_level_rewards
        train_batch.non_tensor_batch: 'solution', 'data_source', 'ability', 'reward_model', 'extra_info', 'question', 'index', 'tools_kwargs', 'interaction_kwargs', 'uid', 'score', 'acc', 'pred'
        test_batch.batch: attention_mask, input_ids, position_ids, responses, prompts 
        test_batch.non_tensor_batch: data_source, reward_model, uid, extra_info, interaction_kwargs, tools_kwargs, ability, index
        """
        
        if len(base_batch) == 0:
            return None, None, None, None
                
        def extract_raw_problem(text: str) -> str | None:
            """
            Extract the math problem part from a prompt string.
            Returns the extracted problem as a string, or None if not found.
            """
            import re
            TEMPLATE_PATTERN = re.compile(
                        r"Solve the following math problem step by step\."
                        r" The last line of your response should be of the form Answer: \$Answer \(without quotes\) where \$Answer is the answer to the problem\."
                        r"\s*(.*?)\s*"
                        r"Remember to put your answer on its own line after \"Answer:\".",
                        re.DOTALL
                        )
            match = TEMPLATE_PATTERN.search(text)
            if not match:
                return None
            return match.group(1).strip()

        prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in base_batch.batch["prompts"]]
        responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in base_batch.batch["responses"]]
        raw_problems = []
        extra_info_list = base_batch.non_tensor_batch.get("extra_info", [])
        for idx, prompt in enumerate(prompts):
            extra_info = extra_info_list[idx]
            if extra_info.get("raw_problem") is None:
                raw_problems.append(extract_raw_problem(prompt))
            else:
                raw_problems.append(extra_info["raw_problem"]) # TODO: ours check if it works for training data.

        response_mask = base_batch.batch.get("response_mask")
        if response_mask is None:
            response_mask = compute_response_mask(base_batch)
        response_lengths = response_mask.sum(dim=-1).cpu().numpy()

        data_source_values = base_batch.non_tensor_batch.get("data_source")
        if data_source_values is None:
            data_source_values = np.array(["unknown"] * len(base_batch), dtype=object)
        else:
            data_source_values = np.asarray(data_source_values, dtype=object)

        if reward_result is not None and reward_result.get("reward_tensor") is not None:
            initial_score_tensor = reward_result["reward_tensor"]
        elif "token_level_scores" in base_batch.batch.keys():
            initial_score_tensor = base_batch.batch["token_level_scores"]
        elif "token_level_rewards" in base_batch.batch.keys():
            initial_score_tensor = base_batch.batch["token_level_rewards"]
        else:
            initial_score_tensor = None

        if initial_score_tensor is None:
            raise ValueError("Cannot find reward tensor for constructing second turn inputs.")

        initial_scores = initial_score_tensor.detach().sum(-1).cpu().numpy()

        if reward_result is not None:
            reward_extra_info = reward_result.get("reward_extra_info", {})
        else:
            reward_extra_info = {}

        if "acc" in reward_extra_info:
            initial_correct = np.asarray(reward_extra_info["acc"], dtype=bool)
        elif "acc" in base_batch.non_tensor_batch:
            initial_correct = np.asarray(base_batch.non_tensor_batch["acc"], dtype=bool)
        else:
            initial_correct = initial_scores > 0

        if "pred" in reward_extra_info and reward_extra_info.get("pred") is not None:
            initial_pred = np.asarray(reward_extra_info["pred"], dtype=object)
        elif "pred" in base_batch.non_tensor_batch:
            initial_pred = np.asarray(base_batch.non_tensor_batch["pred"], dtype=object)
        else:
            initial_pred = None

        uids = base_batch.non_tensor_batch.get("uid")
        if uids is None:
            raise NotImplementedError

        partner_mode = getattr(self.config.self_debate, "partner_mode", "")
        if partner_mode not in ("frequency", "random"):
            raise ValueError(f"Unknown partner_mode: {partner_mode}")
        if meta_info is not None and meta_info.get("validate"):
            debate_repeat = int(getattr(self.config.actor_rollout_ref.rollout.val_kwargs, "n", 1))
        else:
            debate_repeat = int(getattr(self.config.self_debate, "debate_n", 1))

        is_validate = bool(meta_info and meta_info.get("validate"))

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid_to_indices[str(uid)].append(idx)

        num_debate_prompt = getattr(self.config.self_debate, "num_debate_prompt", 0)
        if num_debate_prompt is None:
            num_debate_prompt = 0
        num_debate_prompt = int(num_debate_prompt)
        selected_uids = None
        if num_debate_prompt > 0 and num_debate_prompt < len(uid_to_indices):
            uid_keys = list(uid_to_indices.keys())
            selected_uids = set(np.random.choice(uid_keys, size=num_debate_prompt, replace=False).tolist())

        selected_indices: list[int] = []
        second_turn_prompts: list[str] = []
        initial_response_texts: list[str] = []
        partner_response_texts: list[str] = []
        initial_acc_flags: list[bool] = []
        partner_acc_flags: list[bool] = []
        initial_selected_scores: list[float] = []
        partner_selected_scores: list[float] = []

        def append_pair(idx: int, partner_idx: int):
            problem = raw_problems[idx]
            initial_response = responses[idx].strip()
            partner_response = responses[partner_idx].strip()
            sanitized_initial = initial_response.replace('"', '\\"')
            sanitized_partner = partner_response.replace('"', '\\"')
            second_turn_user_content = SECOND_TURN_USER_PROMPT_TEMPLATE.format(
                partner_response=sanitized_partner,
            )

            second_turn_message = [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem step by step. "
                        "The last line of your response should be of the form Answer: $Answer (without quotes) "
                        'where $Answer is the answer to the problem.\n\n'
                        f"{problem}\n\nRemember to put your answer on its own line after \"Answer:\"."
                    ),
                },
                {"role": "assistant", "content": sanitized_initial},
                {"role": "user", "content": second_turn_user_content},
            ]

            second_turn_prompt = self.tokenizer.apply_chat_template(
                second_turn_message,
                add_generation_prompt=True,
                tokenize=False,
            ) # extra <|im_start|> and <|im_end|> compare to base_prompt
            #TODO ours: remove <|im_start|> and <|im_end|> ?

            second_turn_prompts.append(second_turn_prompt)
            selected_indices.append(idx)
            initial_response_texts.append(initial_response)
            partner_response_texts.append(partner_response)
            initial_acc_flags.append(bool(initial_correct[idx]))
            partner_acc_flags.append(bool(initial_correct[partner_idx]))
            initial_selected_scores.append(float(initial_scores[idx]))
            partner_selected_scores.append(float(initial_scores[partner_idx]))

        for uid_key, group_indices in uid_to_indices.items():
            if selected_uids is not None and uid_key not in selected_uids:
                continue
            valid_indices = [
                idx
                for idx in group_indices
                if max_initial_response_length is None or response_lengths[idx] <= max_initial_response_length
            ]
            if not valid_indices:
                continue

            if partner_mode == "frequency":
                if initial_pred is None:
                    continue

                solution_counter = Counter(
                    initial_pred[idx] for idx in valid_indices if initial_pred[idx] is not None
                )
                if len(solution_counter) < 2:
                    continue

                most_common_solutions = solution_counter.most_common()
                majority_solution = most_common_solutions[0][0]
                second_solution = most_common_solutions[1][0]

                majority_indices = [idx for idx in valid_indices if initial_pred[idx] == majority_solution]
                secondary_indices = [idx for idx in valid_indices if initial_pred[idx] == second_solution]
                if not majority_indices or not secondary_indices:
                    continue

                idx = int(np.random.choice(majority_indices))
                partner_idx = int(np.random.choice(secondary_indices))
                if not is_validate:
                    if np.random.rand() < 0.5:
                        append_pair(idx, partner_idx)
                    else:
                        append_pair(partner_idx, idx)
                else:
                    append_pair(idx, partner_idx)

            elif partner_mode == "random":
                if len(valid_indices) < 2:
                    continue
                idx, partner_idx = np.random.choice(valid_indices, size=2, replace=False)
                append_pair(int(idx), int(partner_idx))

        if debate_repeat > 1:
            def _repeat_list(items: list):
                return [item for item in items for _ in range(debate_repeat)]

            second_turn_prompts = _repeat_list(second_turn_prompts)
            selected_indices = _repeat_list(selected_indices)
            initial_response_texts = _repeat_list(initial_response_texts)
            partner_response_texts = _repeat_list(partner_response_texts)
            initial_acc_flags = _repeat_list(initial_acc_flags)
            partner_acc_flags = _repeat_list(partner_acc_flags)
            initial_selected_scores = _repeat_list(initial_selected_scores)
            partner_selected_scores = _repeat_list(partner_selected_scores)

        if not second_turn_prompts:
            return None, None, None, None

        second_turn_non_tensor: dict[str, np.ndarray] = {}
        for key, arr in base_batch.non_tensor_batch.items():
            dtype = arr.dtype if isinstance(arr, np.ndarray) else object
            if dtype == np.dtype("O"):
                dtype = object
            values = [deepcopy(arr[idx]) for idx in selected_indices]
            second_turn_non_tensor[key] = np.array(values, dtype=dtype)

        second_turn_non_tensor["__num_turns__"] = np.full(len(selected_indices), 2, dtype=np.int32)

        if include_extra_metadata:
            second_turn_non_tensor["debate_initial_response"] = np.array(initial_response_texts, dtype=object)
            second_turn_non_tensor["debate_partner_response"] = np.array(partner_response_texts, dtype=object)
            second_turn_non_tensor["debate_initial_reward"] = np.array(initial_selected_scores, dtype=np.float32)
            second_turn_non_tensor["debate_partner_reward"] = np.array(partner_selected_scores, dtype=np.float32)

        # Keep correctness flags regardless of metadata setting for downstream stats.
        second_turn_non_tensor["debate_initial_correct"] = np.array(initial_acc_flags, dtype=bool)
        second_turn_non_tensor["debate_partner_correct"] = np.array(partner_acc_flags, dtype=bool)

        tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "right")
        tokenizer_kwargs = tokenizer_kwargs or {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.config.data.max_response_length,
        }

        try:
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(second_turn_prompts, **tokenizer_kwargs)
        finally:
            self.tokenizer.padding_side = tokenizer_padding_side

        attention_mask = encoded["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)

        allowed_non_tensor_keys = {
            "tools_kwargs",
            "ability",
            "index",
            "interaction_kwargs",
            "__num_turns__",
            "debate_initial_correct",
            "debate_partner_correct",
        }
        debate_gen_non_tensor = {
            key: val for key, val in second_turn_non_tensor.items() if key in allowed_non_tensor_keys
        }

        debate_gen_batch = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors=debate_gen_non_tensor,
            meta_info=deepcopy(meta_info) if meta_info is not None else {},
        )

        aux_info = {
            "raw_problems": raw_problems,
            "data_source_values": data_source_values,
            "initial_responses": initial_response_texts,
            "partner_responses": partner_response_texts,
            "initial_acc_flags": initial_acc_flags,
            "partner_acc_flags": partner_acc_flags,
            "initial_scores": initial_selected_scores,
            "partner_scores": partner_selected_scores,
        }

        # training
        # debate_gen_batch.batch: input_ids, attention_mask, position_ids
        # debate_gen_batch.non_tensor_batch: tools_kwargs, ability, index, interaction_kwargs

        return debate_gen_batch, second_turn_non_tensor, selected_indices, aux_info

    def _generate_training_second_turn_batch(self, batch: DataProto, metrics: dict, timing_raw: dict):
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.config.data.max_response_length,
        }
        debate_meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.get("do_sample", True),
            "validate": False,
            "global_steps": self.global_steps,
            "response_length": self.config.data.max_response_length,
        }

        debate_gen_batch, second_turn_non_tensor, selected_indices, _ = self._construct_second_turn_inputs(
            batch,
            include_extra_metadata=False,
            tokenizer_kwargs=tokenizer_kwargs,
            meta_info=debate_meta_info,
            max_initial_response_length=self.config.self_debate.train_debate_max_initial_response_length,
        )
        if debate_gen_batch is None or second_turn_non_tensor is None or selected_indices is None:
            return None

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        debate_gen_batch_padded, pad_size = pad_dataproto_to_divisor(debate_gen_batch, size_divisor)

        with marked_timer("debate_gen", timing_raw, "blue"):
            if not self.async_rollout_mode:
                debate_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(debate_gen_batch_padded)
            else:
                debate_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(debate_gen_batch_padded)
            debate_output_gen_batch = unpad_dataproto(debate_output_gen_batch_padded, pad_size=pad_size)

        # debate_output_gen_batch.batch: input_ids, attention_mask, position_ids, responses
        # debate_output_gen_batch.non_tensor_batch: tools_kwargs, ability, index, interaction_kwargs

        debate_output_gen_batch.non_tensor_batch = second_turn_non_tensor
        debate_output_gen_batch.meta_info["validate"] = False

        if self.config.algorithm.use_kl_in_reward:
            debate_output_gen_batch = self.compute_kl_related_metrics(debate_output_gen_batch, metrics, timing_raw)

        if self.use_rm and "rm_scores" not in debate_output_gen_batch.batch.keys():
            reward_tensor = self.rm_wg.compute_rm_score(debate_output_gen_batch)
            debate_output_gen_batch = debate_output_gen_batch.union(reward_tensor)

        first_turn_acc = None
        partner_acc = None
        if "acc" in debate_output_gen_batch.non_tensor_batch:
            first_turn_acc = np.asarray(debate_output_gen_batch.non_tensor_batch["acc"], dtype=bool)
        if "debate_partner_correct" in debate_output_gen_batch.non_tensor_batch:
            partner_acc = np.asarray(debate_output_gen_batch.non_tensor_batch["debate_partner_correct"], dtype=bool)

        before_filter_samples = len(debate_output_gen_batch)
        debate_reward_tensor, debate_reward_extra_infos = compute_reward(debate_output_gen_batch, self.reward_fn)
        debate_output_gen_batch.batch["token_level_scores"] = debate_reward_tensor

        if debate_reward_extra_infos:
            debate_output_gen_batch.non_tensor_batch.update(
                {k: np.array(v) for k, v in debate_reward_extra_infos.items()}
            )

        uid_arr = debate_output_gen_batch.non_tensor_batch.get("uid")
        uid_arr_np = np.asarray(uid_arr) if uid_arr is not None else None

        if uid_arr_np is not None:
            metrics["train/debate_prompts_before_filter"] = len(np.unique(uid_arr_np))
        metrics["train/debate_samples_before_filter"] = before_filter_samples

        def _record_confusion(prefix: str, first: np.ndarray | None, partner: np.ndarray | None, final: np.ndarray | None):
            if first is None or final is None:
                return
            if len(first) != len(final):
                return
            first = np.asarray(first, dtype=bool)
            final = np.asarray(final, dtype=bool)
            if partner is not None and len(partner) == len(final):
                partner = np.asarray(partner, dtype=bool)
                metrics[f"{prefix}/first_1_partner_0_final_1"] = int(np.sum(first & (~partner) & final))
                metrics[f"{prefix}/first_1_partner_0_final_0"] = int(np.sum(first & (~partner) & (~final)))
                metrics[f"{prefix}/first_0_partner_1_final_1"] = int(np.sum((~first) & partner & final))
                metrics[f"{prefix}/first_0_partner_1_final_0"] = int(np.sum((~first) & partner & (~final)))
                metrics[f"{prefix}/first_0_partner_0_final_1"] = int(np.sum((~first) & (~partner) & final))
                metrics[f"{prefix}/first_0_partner_0_final_0"] = int(np.sum((~first) & (~partner) & (~final)))
            else:
                metrics[f"{prefix}/first_1_final_1"] = int(np.sum(first & final))
                metrics[f"{prefix}/first_1_final_0"] = int(np.sum(first & (~final)))
                metrics[f"{prefix}/first_0_final_1"] = int(np.sum((~first) & final))
                metrics[f"{prefix}/first_0_final_0"] = int(np.sum((~first) & (~final)))

        second_turn_acc_before = None
        if "acc" in debate_output_gen_batch.non_tensor_batch:
            second_turn_acc_before = np.asarray(debate_output_gen_batch.non_tensor_batch["acc"], dtype=bool)
            metrics["train/debate_accuracy_before_filter"] = float(second_turn_acc_before.mean())
            if first_turn_acc is not None and len(first_turn_acc) == len(second_turn_acc_before):
                metrics["train/debate_accuracy_improvement_before_filter"] = float(
                    second_turn_acc_before.mean() - first_turn_acc.mean()
                )
                _record_confusion(
                    "train/debate_confusion_before_filter", first_turn_acc, partner_acc, second_turn_acc_before
                )

        if self.config.algorithm.use_kl_in_reward:
            debate_output_gen_batch, kl_metrics = apply_kl_penalty(
                debate_output_gen_batch,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            debate_output_gen_batch.batch["token_level_rewards"] = debate_output_gen_batch.batch["token_level_scores"]

        # Drop uids whose second-turn responses all share the same reward (including single-response groups).
        keep_indices = list(range(len(debate_output_gen_batch)))
        if uid_arr is not None and len(debate_output_gen_batch) > 0:
            reward_sums = debate_output_gen_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
            uid2scores: dict[str, list[float]] = defaultdict(list)
            for uid_val, score_val in zip(uid_arr_np, reward_sums, strict=True):
                uid2scores[str(uid_val)].append(float(score_val))

            def _all_equal(vals: list[float]) -> bool:
                if len(vals) <= 1:
                    return True
                first = vals[0]
                return all(np.isclose(first, v) for v in vals[1:])

            keep_indices = [idx for idx, uid_val in enumerate(uid_arr_np) if not _all_equal(uid2scores[str(uid_val)])]
            if len(keep_indices) != len(debate_output_gen_batch):
                before_len = len(debate_output_gen_batch)
                debate_output_gen_batch = debate_output_gen_batch[keep_indices] if keep_indices else debate_output_gen_batch[:0]
                metrics["train/debate_filtered_samples"] = before_len - len(debate_output_gen_batch)

        metrics["train/debate_samples_after_filter"] = len(debate_output_gen_batch)

        if len(debate_output_gen_batch) == 0:
            metrics["train/debate_prompts_after_filter"] = 0
            return batch

        # After filtering metrics.
        uid_arr_after = debate_output_gen_batch.non_tensor_batch.get("uid")
        if uid_arr_after is not None:
            metrics["train/debate_prompts_after_filter"] = len(np.unique(uid_arr_after))

        if "acc" in debate_output_gen_batch.non_tensor_batch:
            second_turn_acc_after = np.asarray(debate_output_gen_batch.non_tensor_batch["acc"], dtype=bool)
            metrics["train/debate_accuracy_after_filter"] = float(second_turn_acc_after.mean())
            if first_turn_acc is not None:
                first_acc_filtered = np.asarray(first_turn_acc)[keep_indices] if keep_indices else np.asarray(first_turn_acc)
                partner_acc_filtered = (
                    np.asarray(partner_acc)[keep_indices] if partner_acc is not None and keep_indices else partner_acc
                )
                if len(first_acc_filtered) == len(second_turn_acc_after):
                    metrics["train/debate_accuracy_improvement_after_filter"] = float(
                        second_turn_acc_after.mean() - first_acc_filtered.mean()
                    )
                    _record_confusion(
                        "train/debate_confusion_after_filter",
                        first_acc_filtered,
                        partner_acc_filtered,
                        second_turn_acc_after,
                    )

        # Ensure first-turn batch carries debate correctness flags so that
        # non_tensor keys match the debate batch.
        if "debate_initial_correct" not in batch.non_tensor_batch:
            base_initial_correct = batch.non_tensor_batch.get("acc")
            base_initial_correct = np.asarray(base_initial_correct, dtype=bool)
            batch.non_tensor_batch["debate_initial_correct"] = base_initial_correct
        if "debate_partner_correct" not in batch.non_tensor_batch:
            batch.non_tensor_batch["debate_partner_correct"] = np.zeros(len(batch), dtype=bool)

        # Align first-turn and debate batches before concatenation. Second-turn
        # prompts are padded to their own batch max, which can be shorter than
        # the fixed first-turn prompt length.
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def _pad_batch_key(data: DataProto, key: str, target_len: int, pad_val: int):
            if key not in data.batch:
                return
            if data.batch[key].shape[-1] == target_len:
                return
            data.batch[key] = pad_sequence_to_length(
                data.batch[key], max_seq_len=target_len, pad_token_id=pad_val, left_pad=True
            )

        pad_specs = {
            "attention_mask": 0,
            "input_ids": pad_token_id,
            "position_ids": 0,
            "prompts": pad_token_id,
        }
        for key, pad_val in pad_specs.items():
            if key not in batch.batch or key not in debate_output_gen_batch.batch:
                continue
            target_len = max(batch.batch[key].shape[-1], debate_output_gen_batch.batch[key].shape[-1])
            _pad_batch_key(batch, key, target_len, pad_val)
            _pad_batch_key(debate_output_gen_batch, key, target_len, pad_val)

        return DataProto.concat([batch, debate_output_gen_batch])

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        debate_reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            # test_data.keys() = ['input_ids', 'attention_mask', 'position_ids', 'data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids', 'index', 'tools_kwargs', 'interaction_kwargs']
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            uids = test_batch.non_tensor_batch.get("uid")
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            try:
                print("\n", test_gen_batch_padded.batch, "\n")
                if not self.async_rollout_mode: # not False
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            except RuntimeError as exc:
                error_msg = f" Error: {exc}"
                print(error_msg)
                log_dir = getattr(self.config.trainer, "default_local_dir", None) or os.getcwd()
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "validation_oom.log")
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(f"{timestamp} | step={self.global_steps} | {error_msg} \n {test_gen_batch_padded.batch}\n")
                except Exception as log_exc:
                    print(f"[validate] Failed to write OOM log: {log_exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except RayActorError as exc:
                error_msg = f"[validate] Ray actor died during rollout generation; skipping validation. Error: {exc}"
                print(error_msg)
                log_dir = getattr(self.config.trainer, "default_local_dir", None) or os.getcwd()
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "validation_oom.log")
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(
                            f"{timestamp} | step={self.global_steps} | {error_msg} \n {test_gen_batch_padded.batch}\n"
                        )
                except Exception as log_exc:
                    print(f"[validate] Failed to write validation log: {log_exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {}

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store original inputs
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(uids)
            sample_gts.extend(ground_truths)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            # self-debate generation
            debate_reward_extra_infos_dict = self._validate_debate_generation(
                test_batch=test_batch,
                debate_reward_extra_infos_dict=debate_reward_extra_infos_dict,
                reward_result=result,
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:  # None
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "pass"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        debate_second_turn_samples = debate_reward_extra_infos_dict.get("debate_second_turn_samples", [])
        if debate_second_turn_samples:

            def _compute_second_turn_stats(rows: list[dict]) -> dict:
                final_acc_array = np.array(
                    [bool(row.get("second_turn_correct", False)) for row in rows],
                    dtype=bool,
                )
                initial_acc_array = np.array(
                    [bool(row.get("initial_correct", False)) for row in rows],
                    dtype=bool,
                )
                partner_acc_array = np.array(
                    [bool(row.get("partner_correct", False)) for row in rows],
                    dtype=bool,
                )

                second_turn_avg_acc = float(final_acc_array.mean()) if final_acc_array.size else 0.0
                initial_avg_acc = float(initial_acc_array.mean()) if initial_acc_array.size else 0.0
                accuracy_improvement = second_turn_avg_acc - initial_avg_acc

                confusion_matrix = {
                    "first_1_partner_0_final_1": int(
                        np.sum(initial_acc_array & (~partner_acc_array) & final_acc_array)
                    ),
                    "first_1_partner_0_final_0": int(
                        np.sum(initial_acc_array & (~partner_acc_array) & (~final_acc_array))
                    ),
                    "first_0_partner_1_final_1": int(
                        np.sum((~initial_acc_array) & partner_acc_array & final_acc_array)
                    ),
                    "first_0_partner_1_final_0": int(
                        np.sum((~initial_acc_array) & partner_acc_array & (~final_acc_array))
                    ),
                    "first_0_partner_0_final_1": int(
                        np.sum((~initial_acc_array) & (~partner_acc_array) & final_acc_array)
                    ),
                    "first_0_partner_0_final_0": int(
                        np.sum((~initial_acc_array) & (~partner_acc_array) & (~final_acc_array))
                    ),
                }

                return {
                    "second_turn_average_accuracy": second_turn_avg_acc,
                    "initial_average_accuracy": initial_avg_acc,
                    "accuracy_improvement": accuracy_improvement,
                    "confusion_matrix": confusion_matrix,
                    "num_second_turn_samples": len(rows),
                }

            per_source_samples: dict[str, list[dict]] = defaultdict(list)
            for row in debate_second_turn_samples:
                data_source = row.get("data_source", "unknown")
                per_source_samples[str(data_source)].append(row)

            debate_summary: dict[str, dict] = {}
            for data_source, rows in per_source_samples.items():
                source_summary = _compute_second_turn_stats(rows)
                debate_summary[data_source] = source_summary
                prefix = f"val-debate/{data_source}"
                metric_dict[f"{prefix}/average_accuracy"] = source_summary["second_turn_average_accuracy"]
                metric_dict[f"{prefix}/initial_accuracy"] = source_summary["initial_average_accuracy"]
                metric_dict[f"{prefix}/accuracy_improvement"] = source_summary["accuracy_improvement"]
                metric_dict[f"{prefix}/num_samples"] = source_summary["num_second_turn_samples"]
                metric_dict[f"{prefix}/num_second_turn_prompts"] = len(set(rows_i["uid"] for rows_i in rows))
                for cname, cval in source_summary["confusion_matrix"].items():
                    metric_dict[f"{prefix}/confusion/{cname}"] = cval

            print("\n[Debate Validation] Second-turn accuracy summary:")
            pprint(debate_summary)

            debate_reward_extra_infos_dict["debate_second_turn_summary"] = debate_summary

        return metric_dict

    def _validate_debate_generation(self, test_batch, debate_reward_extra_infos_dict, reward_result):
        """
        test_batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']
        test_batch.non_tensor_batch.keys(): ['data_source', 'reward_model', 'extra_info', 'uid', 'tools_kwargs', 'index', 'interaction_kwargs', 'ability']
        decode test_batch.batch["input_ids"] contains both prompt and response : 'system\nYou are a helpful assistant.\nuser\nSolve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nFind the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.\n\nRemember to put your answer on its own line after "Answer:".\nassistant\nAnswer: 462'
        test_batch.batch["prompts"].size() = [batch, max_prompt_size], test_batch.batch["input_ids"].size() = [batch, max_length]
        """ 

        if not getattr(self.config.self_debate, "enable_second_turn_eval", True):
            return debate_reward_extra_infos_dict

        if len(test_batch) == 0:
            return debate_reward_extra_infos_dict

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.config.data.max_response_length,
        }
         #TODO ours: reduce the max response length for efficiency？May not need it in validation.

        debate_meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "global_steps": self.global_steps,
        }

        test_max_initial_response_length = getattr(
            self.config.self_debate, "test_debate_max_initial_response_length", None
        )
        if test_max_initial_response_length is not None and test_max_initial_response_length <= 0:
            test_max_initial_response_length = None

        debate_gen_batch, second_turn_non_tensor, selected_indices, aux_info = self._construct_second_turn_inputs(
            test_batch,
            include_extra_metadata=True,
            tokenizer_kwargs=tokenizer_kwargs,
            meta_info=debate_meta_info,
            reward_result=reward_result,
            max_initial_response_length=test_max_initial_response_length,
        )

        if debate_gen_batch is None or second_turn_non_tensor is None or selected_indices is None:
            return debate_reward_extra_infos_dict

        data_source_values = aux_info["data_source_values"]
        raw_problems = aux_info["raw_problems"]
        initial_response_texts = aux_info["initial_responses"]
        partner_response_texts = aux_info["partner_responses"]
        initial_acc_flags = aux_info["initial_acc_flags"]
        partner_acc_flags = aux_info["partner_acc_flags"]
        initial_selected_scores = aux_info["initial_scores"]
        partner_selected_scores = aux_info["partner_scores"]

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        debate_gen_batch_padded, pad_size = pad_dataproto_to_divisor(debate_gen_batch, size_divisor)
        #TODO ours: reduce the max response length for efficiency？May not need it in validation.
        if not self.async_rollout_mode:
            debate_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(debate_gen_batch_padded)
        else:
            debate_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(debate_gen_batch_padded)
        debate_output_gen_batch = unpad_dataproto(debate_output_gen_batch_padded, pad_size=pad_size)

        # Get second turn responses.
        second_turn_response_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in debate_output_gen_batch.batch["responses"]
        ]
        second_turn_non_tensor["debate_second_turn_text"] = np.array(second_turn_response_texts, dtype=object)

        second_turn_stat_rows = []
        for local_idx, original_idx in enumerate(selected_indices):
            sample_payload = {
                "sample_index": int(original_idx),
                "raw_problem": raw_problems[original_idx],
                "initial_response": initial_response_texts[local_idx],
                "partner_response": partner_response_texts[local_idx],
                "initial_correct": bool(initial_acc_flags[local_idx]),
                "partner_correct": bool(partner_acc_flags[local_idx]),
                "uid": second_turn_non_tensor["uid"][local_idx],
            }
            data_source_val = data_source_values[original_idx] if len(data_source_values) else "unknown"
            sample_payload["data_source"] = str(data_source_val)
            sample_payload["second_turn_response"] = second_turn_response_texts[local_idx]
            second_turn_stat_rows.append(sample_payload)

        debate_output_gen_batch.non_tensor_batch = second_turn_non_tensor
        debate_output_gen_batch.meta_info["validate"] = True

        debate_result = self.val_reward_fn(debate_output_gen_batch, return_dict=True)
        debate_reward_tensor = debate_result["reward_tensor"]
        debate_scores = debate_reward_tensor.sum(-1).cpu().tolist()
        debate_reward_extra_info = debate_result.get("reward_extra_info", {})
        final_acc = debate_reward_extra_info.get("acc")
        final_acc = [bool(flag) for flag in final_acc]

        debate_reward_extra_infos_dict["reward"].extend(debate_scores)
        debate_reward_extra_infos_dict["initial_reward"].extend(initial_selected_scores)
        debate_reward_extra_infos_dict["partner_reward"].extend(partner_selected_scores)
        debate_reward_extra_infos_dict["initial_acc"].extend([bool(x) for x in initial_acc_flags])
        debate_reward_extra_infos_dict["partner_acc"].extend([bool(x) for x in partner_acc_flags])

        if debate_reward_extra_info:
            for key, lst in debate_reward_extra_info.items():
                debate_reward_extra_infos_dict[key].extend(lst)

        debate_reward_extra_infos_dict["debate_improved"].extend(
            [final and not init for final, init in zip(final_acc, initial_acc_flags)]
        )
        debate_reward_extra_infos_dict["debate_regressed"].extend(
            [init and not final for final, init in zip(final_acc, initial_acc_flags)]
        )

        for row, final_flag, score in zip(second_turn_stat_rows, final_acc, debate_scores):
            row["second_turn_correct"] = bool(final_flag)
            row["debate_score"] = float(score)

        debate_reward_extra_infos_dict["debate_second_turn_samples"].extend(second_turn_stat_rows)
        return debate_reward_extra_infos_dict
