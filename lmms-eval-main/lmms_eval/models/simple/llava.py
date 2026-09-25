import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import os
import warnings
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


def _fallback_get_model_name_from_path(model_path: str) -> str:
    """Matches llava.mm_utils.get_model_name_from_path when LLaVA imports fail mid-stack."""
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    return model_paths[-1]


try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)
    get_model_name_from_path = _fallback_get_model_name_from_path

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


def _get_inner_llama_model(model):
    """Get the inner LlamaModel that has _last_prefill_layer_skip_count (LLaVA/LLaVA-NeXT)."""
    if model is None:
        return None
    m = model
    if hasattr(m, "get_model"):
        m = m.get_model()
    elif hasattr(m, "model"):
        m = m.model
    return m


def _record_layer_skip_count(model, out_list: List[int]) -> None:
    """Read _last_prefill_layer_skip_count from inner model and append to out_list."""
    inner = _get_inner_llama_model(model)
    if inner is None:
        return
    count = getattr(inner, "_last_prefill_layer_skip_count", None)
    if count is not None:
        out_list.append(int(count))


def _record_layer_skip_per_layer(model, out_dict: Dict[int, int]) -> None:
    """Read _last_prefill_skipped_layer_indices from inner model and accumulate per-layer counts."""
    inner = _get_inner_llama_model(model)
    if inner is None:
        return
    indices = getattr(inner, "_last_prefill_skipped_layer_indices", None)
    if indices is not None:
        for layer_idx in indices:
            out_dict[layer_idx] = out_dict.get(layer_idx, 0) + 1

def _record_attn_skip_count(model, out_list: List[int]) -> None:
    inner = _get_inner_llama_model(model)
    if inner is None:
        return
    count = getattr(inner, "_last_prefill_attn_skip_count", None)
    if count is not None:
        out_list.append(int(count))


def _record_attn_skip_per_layer(model, out_dict: Dict[int, int]) -> None:
    inner = _get_inner_llama_model(model)
    if inner is None:
        return
    indices = getattr(inner, "_last_prefill_attn_skipped_layer_indices", None)
    if indices is not None:
        for layer_idx in indices:
            out_dict[layer_idx] = out_dict.get(layer_idx, 0) + 1


def _record_mlp_skip_count(model, out_list: List[int]) -> None:
    inner = _get_inner_llama_model(model)
    if inner is None:
        return
    count = getattr(inner, "_last_prefill_mlp_skip_count", None)
    if count is not None:
        out_list.append(int(count))


def _record_mlp_skip_per_layer(model, out_dict: Dict[int, int]) -> None:
    inner = _get_inner_llama_model(model)
    if inner is None:
        return
    indices = getattr(inner, "_last_prefill_mlp_skipped_layer_indices", None)
    if indices is not None:
        for layer_idx in indices:
            out_dict[layer_idx] = out_dict.get(layer_idx, 0) + 1


def _log_layer_skip_summary(out_list: List[int], logger) -> None:
    """Log summary of layer skip counts."""
    if not out_list:
        return
    n = len(out_list)
    total = sum(out_list)
    mean_skip = total / n
    logger.info(
        f"[layer_skip] n_samples={n}, total_skips={total}, mean_skips_per_sample={mean_skip:.2f}, min={min(out_list)}, max={max(out_list)}"
    )

def _build_skip_stats(skip_counts: List[int], skip_per_layer: Dict[int, int], num_layers: int, prefix: str) -> dict:
    if not skip_counts:
        return {}

    total = sum(skip_counts)
    n = len(skip_counts)
    mean_ratio = (total / (n * num_layers) * 100) if n and num_layers else 0

    return {
        f"{prefix}_total_prefill_count": n,
        f"{prefix}_total_layers_skipped": total,
        f"{prefix}_mean_layers_skipped_per_prefill": round(total / n, 2) if n else 0,
        f"{prefix}_mean_skip_ratio_pct": round(mean_ratio, 2),
        f"{prefix}_min_layers_skipped": min(skip_counts),
        f"{prefix}_max_layers_skipped": max(skip_counts),
        f"{prefix}_per_layer_skip_count": {
            str(li): skip_per_layer.get(li, 0) for li in range(num_layers)
        },
        f"{prefix}_per_layer_skip_pct": {
            str(li): round(skip_per_layer.get(li, 0) / n * 100, 2) for li in range(num_layers)
        } if n else {},
    }

@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        router_top_k: Optional[int] = None,  # passed to LlavaConfig / modeling_llama skip threshold
        router_weight_path: Optional[str] = None,  # optional safetensors containing router weights
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        if router_top_k is not None:
            llava_model_args["overwrite_config"] = {"router_top_k": int(router_top_k)}
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        if router_weight_path:
            self._load_router_weights(router_weight_path)
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # Accumulate layer skip counts for evaluation stats (when using visual token skipping)
        self._layer_skip_counts: List[int] = []
        self._layer_skip_per_layer: Dict[int, int] = {}  # layer_idx -> count of samples that skipped it
        self._attn_skip_counts: List[int] = []
        self._attn_skip_per_layer: Dict[int, int] = {}
        self._mlp_skip_counts: List[int] = []
        self._mlp_skip_per_layer: Dict[int, int] = {}
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def _load_router_weights(self, router_weight_path: str) -> None:
        if not os.path.isfile(router_weight_path):
            raise FileNotFoundError(f"router_weight_path does not exist: {router_weight_path}")
        state_dict = safe_load_file(router_weight_path)
        if not state_dict:
            raise ValueError(f"router safetensors is empty: {router_weight_path}")

        model_state = self._model.state_dict()
        model_keys = set(model_state.keys())

        # Build suffix lookup so checkpoints with different wrappers/prefixes
        # (e.g. language_model.model.* vs model.*) can still be aligned safely.
        router_suffix_to_model_keys: Dict[str, List[str]] = {}
        for mk in model_keys:
            for anchor in ("attn_routers.", "mlp_routers."):
                pos = mk.find(anchor)
                if pos >= 0:
                    suffix = mk[pos:]
                    router_suffix_to_model_keys.setdefault(suffix, []).append(mk)
                    break

        remapped_state_dict = {}
        remapped_pairs: List[Tuple[str, str]] = []
        ambiguous_source_keys: List[str] = []
        collision_target_keys: List[str] = []

        for src_k, tensor in state_dict.items():
            dst_k = src_k if src_k in model_keys else None
            if dst_k is None:
                for anchor in ("attn_routers.", "mlp_routers."):
                    pos = src_k.find(anchor)
                    if pos < 0:
                        continue
                    suffix = src_k[pos:]
                    candidates = router_suffix_to_model_keys.get(suffix, [])
                    if len(candidates) == 1:
                        dst_k = candidates[0]
                    elif len(candidates) > 1:
                        ambiguous_source_keys.append(src_k)
                    break

            if dst_k is None:
                continue
            if dst_k in remapped_state_dict:
                collision_target_keys.append(dst_k)
                continue
            remapped_state_dict[dst_k] = tensor
            if dst_k != src_k:
                remapped_pairs.append((src_k, dst_k))

        matched_keys = list(remapped_state_dict.keys())
        if not matched_keys:
            sample_keys = list(state_dict.keys())[:5]
            raise ValueError(
                f"No keys in {router_weight_path} match model state dict, even after router suffix remap. "
                f"First keys in file: {sample_keys}"
            )

        incompatible = self._model.load_state_dict(remapped_state_dict, strict=False)
        eval_logger.info(
            f"Loaded router weights from {router_weight_path}: "
            f"matched={len(matched_keys)}, missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        if remapped_pairs:
            preview = ", ".join([f"{src} -> {dst}" for src, dst in remapped_pairs[:3]])
            eval_logger.info(
                f"Router key remap applied ({len(remapped_pairs)} keys). "
                f"Examples: {preview}"
            )
        if ambiguous_source_keys:
            eval_logger.warning(
                f"Skipped {len(ambiguous_source_keys)} ambiguous router keys while remapping. "
                f"First key: {ambiguous_source_keys[0]}"
            )
        if collision_target_keys:
            eval_logger.warning(
                f"Skipped {len(collision_target_keys)} colliding remapped keys. "
                f"First target: {collision_target_keys[0]}"
            )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # Record layer skip counts for evaluation stats.
            _record_layer_skip_count(self.model, self._layer_skip_counts)
            _record_layer_skip_per_layer(self.model, self._layer_skip_per_layer)
            _record_attn_skip_count(self.model, self._attn_skip_counts)
            _record_attn_skip_per_layer(self.model, self._attn_skip_per_layer)
            _record_mlp_skip_count(self.model, self._mlp_skip_counts)
            _record_mlp_skip_per_layer(self.model, self._mlp_skip_per_layer)
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        _log_layer_skip_summary(self._layer_skip_counts, eval_logger)
        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            # prompts_input = contexts[0]

            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            try:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                # Record layer skip counts from modeling_llama.
                _record_layer_skip_count(self.model, self._layer_skip_counts)
                _record_layer_skip_per_layer(self.model, self._layer_skip_per_layer)
                _record_attn_skip_count(self.model, self._attn_skip_counts)
                _record_attn_skip_per_layer(self.model, self._attn_skip_per_layer)
                _record_mlp_skip_count(self.model, self._mlp_skip_counts)
                _record_mlp_skip_per_layer(self.model, self._mlp_skip_per_layer)
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LMM
            # if self.truncate_context:
            #     cont_toks = cont_toks[input_ids.shape[1] :]
            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            # if self.truncate_context:
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        _log_layer_skip_summary(self._layer_skip_counts, eval_logger)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")

    def get_layer_skip_stats(self) -> Optional[dict]:
        num_layers = getattr(getattr(self, "_config", None), "num_hidden_layers", 32)

        has_any = bool(self._layer_skip_counts) or bool(self._attn_skip_counts) or bool(self._mlp_skip_counts)
        if not has_any:
            return None

        result = {}
        # combined
        if self._layer_skip_counts:
            total = sum(self._layer_skip_counts)
            n = len(self._layer_skip_counts)
            mean_ratio = (total / (n * num_layers) * 100) if n and num_layers else 0
            result.update({
                "total_prefill_count": n,
                "total_layers_skipped": total,
                "mean_layers_skipped_per_prefill": round(total / n, 2) if n else 0,
                "mean_skip_ratio_pct": round(mean_ratio, 2),
                "min_layers_skipped": min(self._layer_skip_counts),
                "max_layers_skipped": max(self._layer_skip_counts),
                "per_layer_skip_count": {
                    str(li): self._layer_skip_per_layer.get(li, 0)
                    for li in range(num_layers)
                },
                "per_layer_skip_pct": {
                    str(li): round(self._layer_skip_per_layer.get(li, 0) / n * 100, 2)
                    for li in range(num_layers)
                } if n else {},
            })

        # attn
        result.update(_build_skip_stats(
            self._attn_skip_counts,
            self._attn_skip_per_layer,
            num_layers,
            "attn"
        ))

        # mlp
        result.update(_build_skip_stats(
            self._mlp_skip_counts,
            self._mlp_skip_per_layer,
            num_layers,
            "mlp"
        ))

        return result
