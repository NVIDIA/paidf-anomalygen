# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from tqdm import tqdm
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torchvision.transforms import Resize
from PIL import Image


from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.conditioner import DataType, T2VCondition
from cosmos_predict2.datasets.utils import IMAGE_RES_SIZE_INFO
from cosmos_predict2.models.text2image_dit import MiniTrainDIT
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.module.denoise_prediction import DenoisePrediction
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.pipelines.base import BasePipeline
from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.utils import log, misc
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.models.ag_modules.adapter import build_adapter
from cosmos_predict2.models.ag_modules.mask_encoder import build_mask_encoder
from cosmos_predict2.models.ag_modules.anomaly_LUT import build_anomaly_LUT
from cosmos_predict2.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states


IS_PREPROCESSED_KEY = "is_preprocessed"


NEG_PROMPT = "The video captures a series of frames showing a product without any anomalies. " \
            "Flawless, Precise, Pure, Reliable, Durable, Uniform, Aesthetic, Seamless, " \
            "Functional, Intact, High-quality, Polished, Accurate, Consistent, Efficient. " \
            "Overall, the product is of high quality."



def sample_batch_image(resolution: str = "1024", batch_size: int = 1) -> dict:
    h, w = IMAGE_RES_SIZE_INFO[resolution]["9,16"]  # type: ignore
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }
    return data_batch


def get_sample_batch(
    resolution: str = "512",
    batch_size: int = 1,
) -> dict:
    data_batch = sample_batch_image(resolution, batch_size)

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch

class AnomalyGenPipeline(Text2ImagePipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

        # Anomaly gen componentss
        ### Precision
        self.ag_precision = None
        self.ad_tensor_kwargs = None

        ### Training parameters
        self.max_adaptive_mask_weight = 100
        self.loss_type = None

        ### Text inversion
        self.placeholder_token_id = 1429 # '*', the placeholder token for text inversion
        self.neg_prompt = NEG_PROMPT
        self.neg_condition, self.neg_attn_mask = None, None # Cache negative condition and prompt since they never change

        ### Anomaly embedding
        self.anomaly_embedding_token_dim = 1024 # Related to text encoder's output dimension
        self.anomaly_embedding_init_word = "abnormal"

        ### Model architecture
        self.adapater = None
        self.mask_encoder = None
        self.anomaly_embedding = None

    @classmethod
    def from_config(
        cls,
        config: LazyDict,
        dit_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model_ema_reg.pt",
        text_encoder_path: str = "checkpoints/google-t5/t5-11b",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> Any:
        # Create a pipe
        pipe = cls(device=device, torch_dtype=torch_dtype)

        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )
        pipe.scaling = RectifiedFlowScaling(pipe.sigma_data, config.rectified_flow_t_scaling_factor)

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert (
            pipe.tokenizer.latent_ch == pipe.config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"

        # 4. Load text encoder
        pipe.text_encoder = CosmosT5TextEncoder(device=device, cache_dir=text_encoder_path)
        pipe.text_encoder.to(device)
        pipe.text_tokenizer = pipe.text_encoder.tokenizer # To avoid naming conflict w/ cosmos tokenizer

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        pipe.conditioner.embedders['text'].is_trainable = True # This is a workaround to enable gradient for text embedding

        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None

        # 6. Load DiT
        assert dit_path is not None, "dit_path must be provided to load the model"
        log.info(f"Loading DiT from {dit_path}")
        with init_weights_on_device():
            dit_config = config.net
            pipe.dit = instantiate(dit_config).eval()  # inference

        state_dict = load_state_dict(dit_path)
        # drop net. prefix
        state_dict_dit_compatible = dict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                state_dict_dit_compatible[k[4:]] = v
            else:
                state_dict_dit_compatible[k] = v
        # pipe.dit.load_state_dict(state_dict, assign=True)
        pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
        del state_dict, state_dict_dit_compatible
        log.success(f"Successfully loaded DiT from {dit_path}")

        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe

    def from_anomaly_gen_config(self, ag_config: LazyDict) -> Any:
        # 1. Set precision
        self.ad_precision = ag_config.ad_precision
        log.info(f"Precision for extra components: {self.ad_precision}")
        if self.ad_precision == 'float32':
            self.ad_precision = torch.float32
        elif self.ad_precision == 'bfloat16':
            self.ad_precision = torch.bfloat16
        else:
            raise ValueError(f"Precision {self.ad_precision} not supported")
        self.ad_tensor_kwargs = {"device": "cuda", "dtype": self.ad_precision}

        # 2. Init mask encoder
        log.info(f"Initialize mask encoder")
        self.mask_encoder = self._initialize_mask_encoder(
            ag_config.mask_encoder.encoder_type,
            ag_config.mask_encoder.encoder_config,
            ag_config.mask_encoder.freeze
        ).to(**self.ad_tensor_kwargs)

        # 3. Init. adapter
        log.info(f"Initialize adapter")
        self.adapter = self._initialize_adapter(
            ag_config.adapter.adapter_type,
            ag_config.adapter.adapter_config,
            ag_config.adapter.freeze
        ).to(**self.ad_tensor_kwargs)

        # 4. Init. anomaly embedding
        log.info(f"Initialize anomaly embedding")
        self.anomaly_embedding = self._initialize_anomaly_embedding(
            ag_config.anomaly_embedding.num_tokens,
            ag_config.anomaly_embedding.anomaly_types,
            self.anomaly_embedding_token_dim,
            self.anomaly_embedding_init_word,
            ag_config.anomaly_embedding.freeze
        ).to(**self.ad_tensor_kwargs)

        # Text tokenizer max length
        self.text_tokenizer_max_length = ag_config.text_tokenizer.max_length

        # 5. Optionally swap to a lighter T5 encoder variant
        t5_model_name = getattr(ag_config, 't5_model_name', None)
        if t5_model_name is not None:
            log.info(f"Replacing T5 encoder with lighter variant: {t5_model_name}")
            self.text_encoder = CosmosT5TextEncoder(
                model_name=t5_model_name, device="cuda", cache_dir=t5_model_name
            )
            self.text_tokenizer = self.text_encoder.tokenizer

        # 6. Move text encoder to the same precision as anomaly gen components
        self.text_encoder.text_encoder.to(**self.ad_tensor_kwargs)

        return

    def _initialize_adapter(self, adapter_type, adapter_config, freeze, **kwargs):
        log.info(f"Now building adapter with {adapter_type}")
        return build_adapter(adapter_type, adapter_config, freeze)

    def _initialize_anomaly_embedding(self, num_tokens, anomaly_types, token_dim = 1024, init_word = None, freeze = False, **kwargs):
        log.info(f"Now building anomaly LUT")
        if init_word is None:
            init_embedding = None
        else:
            log.info(f"Use T5 Encoder to convert {init_word} as anomaly embedding's initial value.")
            initial_token = self.text_tokenizer.encode(init_word)
            assert len(initial_token) == 2, f"Initial word {init_word} is tokenized into >1 tokens. Please change your init_word"
            initial_token = torch.tensor(initial_token[0]).cuda() # Remove <EOS> token
            init_embedding = self.text_encoder.text_encoder.shared(initial_token)
        return  build_anomaly_LUT(num_tokens, anomaly_types, token_dim, init_embedding, freeze)

    def _get_text_embedding(self, text: List):
        tokenized_output = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.text_tokenizer_max_length  # Reduced from 512 to save T5 memory (text ~50 + spatial 57 = ~107)
        )
        input_ids = tokenized_output['input_ids'].cuda()                       # [B, 512]
        attn_mask = tokenized_output['attention_mask'].cuda()        # [B, 512]
        text_embedding = self.text_encoder.text_encoder.shared(input_ids)           # [B, 512, 1024]
        return input_ids, attn_mask, text_embedding

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.dit.fully_shard(mesh=dp_mesh)
        self.dit = fully_shard(self.dit, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.dit, dp_mesh)

        # Shard the AnomalyGen components.
        self.mask_encoder.fully_shard(mesh=dp_mesh)
        self.mask_encoder = fully_shard(
            self.mask_encoder, mesh=dp_mesh, reshard_after_forward=True
        )
        broadcast_dtensor_model_states(self.mask_encoder, dp_mesh)

        # TODO(hongyuc): `fully_shard` on `self.adapter` will cause
        # `RuntimeError: Size mismatch` in `cosmos_predict2/utils/fused_adam_dtensor.py`
        # So we skip it for now.

        # self.anomaly_embedding is not a nn.Module, so we don't shard it.

    def _text_inversion(self, data_batch):
        """
        Prepare anomaly condition via text inversion.
        1) Form spatial anomaly embedding
        2) Text inversion
        3) Forward by Adapter & Conditioner to get final condition
        The output condition is always in bfloat16 precision.
        """
        """ Form spatial anomaly embedding """
        # Prepare mask_embedding 
        mask_embedding = self.mask_encoder(data_batch['mask'].to(self.ad_precision)) # [B, L, 1024]
        B, num_mask_tokens, C = mask_embedding.shape        

        # Prepare anomaly embedding
        input_anomaly_embedding = torch.stack([
            self.anomaly_embedding[anomaly_name]
            for anomaly_name in data_batch['name']
        ]) # [B, #token, 1024]

        # Merge mask embedding & anomaly embedding
        spatial_anomaly_embedding = torch.cat([mask_embedding, input_anomaly_embedding], dim = 1)
        _, total_tokens, _ = spatial_anomaly_embedding.shape

        """ Text inversion """

        # Prepare negative text embedding
        if self.neg_condition is None:
            neg_input_ids, neg_attn_mask, neg_text_embedding = self._get_text_embedding([self.neg_prompt]) # [B, 512]
            # CosmosT5TextEncoder.text_encoder is the actual text encoder
            neg_condition = self.text_encoder.text_encoder(inputs_embeds = neg_text_embedding, 
                                                                                                    attention_mask = neg_attn_mask).last_hidden_state  # type: ignore
            neg_condition = neg_condition.to(torch.bfloat16)
            neg_attn_mask = neg_attn_mask.to(torch.bfloat16)
            neg_condition[0][neg_attn_mask[0] == 0] = 0
            self.neg_condition = neg_condition[0]
            self.neg_attn_mask = neg_attn_mask[0]

        neg_condition = torch.stack([self.neg_condition] * B)
        neg_attn_mask = torch.stack([self.neg_attn_mask] * B)

        # Prepare text embedding
        input_ids, attn_mask, text_embedding = self._get_text_embedding(data_batch['caption'])
        B, L = input_ids.shape

        # Check if placeholder exists exactly once in every prompt
        occurrence = (input_ids == self.placeholder_token_id).sum(dim=1)
        assert (occurrence == 1).all().item(), f"[ERROR] Placeholder token id {self.placeholder_token_id} does not exist exactly once in every prompt!\n"\
                                                                        f"Occurance: {occurrence}"

        # Find the index of placeholder
        placeholder_rows, placeholder_cols = torch.where(input_ids == self.placeholder_token_id)
        sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True) # token index for placeholder
        sorted_rows = placeholder_rows[sort_idx] # Reorder

        # Insert embedding, one data at a time
        for idx in range(B):
            row, col = sorted_rows[idx], sorted_cols[idx]
            num_text_tokens = torch.where(input_ids[row] != 0)[0].max()
            final_length = num_text_tokens + total_tokens - 1            
            assert final_length <= L, f"[ERROR] After text inversion, number of tokens {final_length} exceeds maximum length of text encoder {L}"

            # Form new tokens
            new_token_row = torch.cat(
                [
                    input_ids[row][:col], 
                    torch.tensor(self.placeholder_token_id).repeat(total_tokens).to('cuda'),
                    input_ids[row][col + 1: ]
                ], dim=0)[:L] # Truncation
            input_ids[row] = new_token_row

            # Form new embedding            
            new_embed_row = torch.cat(
            [
                text_embedding[row][:col], 
                spatial_anomaly_embedding[row],
                text_embedding[row][col + 1 : ]
            ], dim=0)[:L] # Truncation
            text_embedding[row] = new_embed_row

            # Update attention mask
            attn_mask[row][:final_length] = True

        """ Forward by Adapter & Conditioner to get final condition """
        # adapter
        text_embedding = self.adapter(text_embedding)

        # T5 embedding
        condition = self.text_encoder.text_encoder(
            inputs_embeds=text_embedding, attention_mask=attn_mask
        ).last_hidden_state

        # Clear masked 
        for idx in range(B):
            condition[idx][attn_mask[idx] == 0] = 0

        # Always set condition to bfloat16
        condition = condition.to(torch.bfloat16)
        attn_mask = attn_mask.to(torch.bfloat16)

        return condition, attn_mask, neg_condition, neg_attn_mask

    def _initialize_mask_encoder(self, encoder_type, encoder_config, freeze, **kwargs):
        log.info(f"Now building mask encoder with {encoder_type}")
        return build_mask_encoder(encoder_type, encoder_config, freeze)

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, T2VCondition]:
        self._augment_image_dim_inplace(data_batch)

        # Latent state
        raw_state = data_batch["images"]
        latent_state = self.encode(raw_state).contiguous().float()

        # Text inversion
        condition, attn_mask, neg_condition, neg_attn_mask = self._text_inversion(data_batch)
        B, L, C = condition.shape
        data_batch['t5_text_embeddings'] = condition
        data_batch['t5_text_mask'] = torch.ones(B, 512, dtype=torch.bfloat16).cuda() # Following original cosmos diffusion's implementation
        data_batch['neg_t5_text_embeddings'] = neg_condition
        data_batch['neg_t5_text_mask'] = torch.ones(B, 512, dtype=torch.bfloat16).cuda() # Following original cosmos diffusion's implementation

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE)
        return raw_state, latent_state, condition

    def _get_guided_mask_and_weight(self, x0_gt, masks, max_adaptive_mask_weight=100):
        """ Retain only un-masked regions via masks"""
        B, C, T, H, W = x0_gt.shape
        # Build guided mask
        guided_masks = torch.zeros((B, 1, H, W))
        for i in range(B):
            latent_mask = 1 - Resize((H, W), interpolation = Image.Resampling.BICUBIC)(masks[i]) # latent mask: 1 for using denoised result, 0 for using guided_image
            latent_mask[latent_mask <= 0.85] = 0
            latent_mask[latent_mask > 0.85] = 1
            guided_masks[i] = 1 - latent_mask  #guided_mask: 0 for using denoised result, 1 for using guided_image
        guided_masks = guided_masks[:, :, None, :, :].to("cuda")
        # Get adaptive weight
        masked_region = guided_masks.sum(axis = (1, 2, 3, 4))  # On batch dimension
        adaptive_weight = (H * W) / masked_region
        adaptive_weight = torch.clamp(adaptive_weight, min=1.0, max=max_adaptive_mask_weight)
        return  guided_masks, adaptive_weight

    @torch.no_grad()
    def __call__(self, data_batch: dict, seed: int = 0, guidance = 4.0,
                           num_steps: int = 35, is_negative_prompt: bool = True, n_sample: int = 2,
                           use_cuda_graphs: bool = False):
        ''' Run inference to generate images from data_batch'''
        # Generate samples
        log.info(f"Begin diffusion fwd process. Guidance = {guidance}, Num steps = {num_steps}, seed = {seed}.")

        # Support per-sample guidance: convert list/tuple to [B,1,1,1,1] tensor
        if isinstance(guidance, (list, tuple)):
            guidance = torch.tensor(guidance, dtype=torch.float32, device=data_batch['images'].device)
            guidance = guidance.view(-1, 1, 1, 1, 1)

        # Prepare guided image & mask (encode expects 5D: B,C,T,H,W)
        images = data_batch['images']
        if images.ndim == 4:
            images = images.unsqueeze(2)
        data_batch['guided_image'] = self.encode(images).contiguous()
        data_batch['guided_mask'], _ = self._get_guided_mask_and_weight(
            x0_gt=data_batch['guided_image'],
            masks=data_batch['mask']
        )
        data_batch['guided_mask'] = 1 - data_batch['guided_mask'] # 1 for using gt, 0 for using denoised result

        # Preprocess
        self._augment_image_dim_inplace(data_batch)
        input_key = "images"
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        # Text inversion
        _, latent_state, _  = self.get_data_and_condition(data_batch)

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # Ensures we are conditioning on IMAGE data type.
        condition = condition.edit_data_type(DataType.IMAGE)
        uncondition = uncondition.edit_data_type(DataType.IMAGE)

        # Context parallelism is disabled for text2image
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, uncondition, None, None)

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_steps, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating image", leave=False)):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            cond_x0 = self.denoise(sample, sigma_in, condition, use_cuda_graphs=use_cuda_graphs).x0
            uncond_x0 = self.denoise(sample, sigma_in, uncondition, use_cuda_graphs=use_cuda_graphs).x0
            x0_pred = cond_x0 + guidance * (cond_x0 - uncond_x0)

            # Replacement trick that enables inpainting with base model
            if "guided_image" in data_batch:
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                x0_pred = guide_mask * guide_image + (1 - guide_mask) * x0_pred

            # Scheduler step (handles float64 internally, returns original dtype)
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])

        # Final clean pass.
        cond_x0 = self.denoise(sample, sigma_in, condition, use_cuda_graphs=use_cuda_graphs).x0
        uncond_x0 = self.denoise(sample, sigma_in, uncondition, use_cuda_graphs=use_cuda_graphs).x0
        samples = cond_x0 + guidance * (cond_x0 - uncond_x0)

        # Replacement trick that enables inpainting with base model
        if "guided_image" in data_batch:
            assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
            guide_image = data_batch["guided_image"]
            guide_mask = data_batch["guided_mask"]
            samples = guide_mask * guide_image + (1 - guide_mask) * samples

        # decode
        image = self.decode(samples)
        log.success("Image generation completed successfully")
        return image
