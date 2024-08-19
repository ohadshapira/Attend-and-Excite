
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
#from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor #noa added 14.8.24

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from diffusers import LCMScheduler, AutoPipelineForText2Image #noa added 14.08.24
#import matplotlib.pyplot as plt #noa added 14.08.24
#from transformers import AutoTokenizer #noa added 14.08.24
#from diffusers import PNDMScheduler #noa added 14.08.24

from utils_project.gaussian_smoothing import GaussianSmoothing #noa update name of utils dir to utils_project, 15.8.24
from utils_project.ptp_utils import AttentionStore, aggregate_attention #noa update name of utils dir to utils_project, 15.8.24


#--------------------------Ohad added 15.8.24 - prompt to dict - start
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from word2number import w2n #noa added 19.8.24 - for prompt converting to dict

# Download necessary nltk data
nltk.download('punkt')
nltk.download('wordnet')
#--------------------------Ohad added 15.8.24 - prompt to dict - end
import matplotlib.pyplot as plt #noa added 19.8.24

logger = logging.get_logger(__name__)

class AttendAndExcitePipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt( #we should keep it (original)- noa 15.8.24
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self, #we should **NOT** keep it (original)- noa 15.8.24
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore, #we should **NOT** keep it (original)- noa 15.8.24
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        return max_attention_per_index


    #--------------------------Ohad added 15.8.24 - prompt to dict - start
    @staticmethod
    def lemmatize_word(word):
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(word, pos='n')


    def count_objects_by_indices_nums(self,prompt, object_indices): #noa - changed func name to _nums - and added a func
        """creates dict from prompt for the number of objects needed for each object"""
        object_indices_offset_corrected=[i-1 for i in object_indices]
        tokens = word_tokenize(prompt.lower())

        counts = defaultdict(int)

        for index in object_indices_offset_corrected:
            word = tokens[index]
            # Look for a number preceding the object (assume it immediately precedes the object)
            number = 1  # Default count
            if index > 0 and tokens[index - 1].isdigit():
                number = int(tokens[index - 1])

            # Convert the word to its singular form
            singular_form = self.lemmatize_word(word)
            counts[singular_form] += number

        return dict(counts)
    #--------------------------Ohad added 15.8.24 - prompt to dict - end

    #----------------------------Noa Added 19.8.24 - prompt to dict with words - START
    def count_objects_by_indices(self, prompt, object_indices):
        """
        Creates a dictionary from the prompt for the number of objects
        needed for each object, using the given indices.
        """
        # Convert the indices to be 0-based (since Python lists are 0-indexed)
        object_indices_offset_corrected = [i-1 for i in object_indices]
        tokens = word_tokenize(prompt.lower())  # Tokenize the prompt into words

        counts = defaultdict(int)  # Dictionary to store the count of each object

        for index in object_indices_offset_corrected:
            word = tokens[index]

            # Look for a number preceding the object (assume it immediately precedes the object)
            if index > 0:
                prev_word = tokens[index - 1]
                try:
                    if prev_word.isdigit():
                        number = int(prev_word)  # Convert digit-based number to integer
                    else:
                        number = w2n.word_to_num(prev_word)  # Convert word-based number to integer
                except ValueError:
                    number = 1  # Default count if the word is not a number
            else:
                number = 1  # Default count if there's no preceding word
            
            # Convert the object word to its singular form
            singular_form = self.lemmatize_word(word)
            counts[singular_form] += number  # Update the object count in the dictionary

        return dict(counts)  # Return the dictionary with object counts
    #----------------------------Noa Added 19.8.24 - prompt to dict with words - START

    #-------------------------Noa added 15.8.24 - detector to dict - start
    # Get the detected object counts from YOLO results
    def object_dict_from_dtector(self,sentence):
        """converts the output of the detector to dictionary of objects and number of objects"""
        detected_counts = {}
        for *box, conf, cls in sentence.xyxy[0]:
            cls_name = sentence.names[int(cls)]
            detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1

        return detected_counts

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor: #we should **NOT** keep it (original)- noa 15.8.24
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    #noa added 18.8.24 -------------------------------------------------------------------------------------------------------------------------------------------------------START 
    def _compute_loss_make_it_count_project2(self, prompt_num_object_main: Dict[str, int], detector_num_object_main: Dict[str, int], return_losses: bool = False) -> torch.Tensor:
        """ Computes the make-it-count-project loss using the maximum L2 distance for each token. """
        losses = []

        for obj in prompt_num_object_main.keys():  # Only consider objects that are in the prompt
            count_prompt = torch.tensor(prompt_num_object_main.get(obj, 0), dtype=torch.float32, requires_grad=True)
            count_detector = torch.tensor(detector_num_object_main.get(obj, 0), dtype=torch.float32, requires_grad=True)
            
            # Compute L2 distance and ensure it's differentiable
            l2_distance = (count_prompt - count_detector) ** 2
            
            # Append loss for each object
            losses.append(l2_distance)
        
        # Combine the losses
        loss = torch.max(torch.stack(losses))  # Use max of losses
        
        if return_losses:
            return loss, losses
        else:
            return loss
          
    def total_variation_loss(self, latents: torch.Tensor):
        """
        Computes the Total Variation (TV) Loss for the input tensor x.
        Args:
            x (torch.Tensor): Latent variable tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Total variation loss.
        """
        loss = torch.mean(torch.abs(latents[:, :, :, :-1] - latents[:, :, :, 1:])) + \
            torch.mean(torch.abs(latents[:, :, :-1, :] - latents[:, :, 1:, :]))
        return loss

    
    def _compute_loss_with_tv(self, prompt_num_object_main: Dict[str, int], detector_num_object_main: Dict[str, int], latents: torch.Tensor, tv_weight: float = 0.1) -> torch.Tensor:
        """
        Computes the combined loss with object count and Total Variation (TV) Loss.
        Args:
            prompt_num_object_main (Dict[str, int]): Dictionary with object counts from the prompt.
            detector_num_object_main (Dict[str, int]): Dictionary with object counts from the detector.
            latents (torch.Tensor): Latent variables tensor.
            tv_weight (float): Weight for the Total Variation Loss.
        Returns:
            torch.Tensor: Combined loss.
        """
        losses = []

        for obj in prompt_num_object_main.keys():
            count_prompt = prompt_num_object_main.get(obj, 0)
            count_detector = detector_num_object_main.get(obj, 0)
            l2_distance = (count_prompt - count_detector) ** 2
            losses.append(l2_distance)

        # Combine losses
        #object_count_loss = max(losses) if losses else torch.tensor(0.0, dtype=torch.float32)
        object_count_loss = self._compute_loss_make_it_count_project2(prompt_num_object_main, detector_num_object_main) #Noa added 19.8.24
        tv_loss = self.total_variation_loss(latents)
        total_loss = object_count_loss + tv_weight * tv_loss

        return total_loss

    #noa added 18.8.24 -------------------------------------------------------------------------------------------------------------------------------------------------------END 

    #noa added 15.8.24 -------------------------------------------------------------------------------------------------------------------------------------------------------START            
    def _compute_loss_make_it_count_project(self, prompt_num_object_main: Dict[str, int], detector_num_object_main: Dict[str, int], return_losses: bool = False) -> torch.Tensor:
        """ Computes the make-it-count-project loss using the maximum L2 distance for each token.  """
        losses = []

        for obj in prompt_num_object_main.keys():# was: 'for obj in all_objects:' # ohad changed: only objects that in prompt are relevant
            count_prompt = prompt_num_object_main.get(obj, 0)
            count_detector = detector_num_object_main.get(obj, 0)
            l2_distance = (count_prompt - count_detector) ** 2
            losses.append(torch.tensor(l2_distance, dtype=torch.float32))
        
        loss = max(losses)       
        if return_losses:
            return loss, losses
        else:
            return loss
        
    def _perform_iterative_refinement_step_make_it_coun_project(self, #we should keep it while making changes or create simialer function (original)- noa 15.8.24
                                           latents: torch.Tensor,
                                           loss: torch.Tensor,
                                           prompt_num_object: Dict[str, int], #noa added 15.8.24
                                           detector_num_object: Dict[str, int], #noa added 15.8.24
                                           text_embeddings: torch.Tensor,
                                           step_size: float,
                                           t: int
                                           ):
        """
        Performs the iterative latent refinement introduced in the project. Here, we continuously update the latent
        code according to our loss objective in each denoising step. 
        """
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # compute loss
        loss, losses = self._compute_loss_make_it_count_project(prompt_num_object_main=prompt_num_object, detector_num_object_main=detector_num_object, return_losses=True)

        if loss != 0:
            latents = self._update_latent(latents, loss, step_size)

        with torch.no_grad():
            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

        try: #remove? noa 15.8.24
            low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
        except Exception as e:
            print(e)  # catch edge case :)
            low_token = np.argmax(losses)

        #low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])

        print(f"\t Finished with loss of: {loss}")
        return loss, latents
    
    #noa added 15.8.24 -------------------------------------------------------------------------------------------------------------------------------------------------------END

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor: #we should keep it (original)- noa 15.8.24
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self, #we should keep it while making changes or create simialer function (original)- noa 15.8.24
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
                )

            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False, #I do not think we need it - Noa 15.8.24 (it is if we want to run the stable diffusion without attend-and-excite changes)
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        #--------------------------noa added 14-15.8.24 - LCM model - start
        # Load the LCM model
        model_lcm_id = "Lykon/dreamshaper-7"
        adapter_lcm_id = "latent-consistency/lcm-lora-sdv1-5"

        pipe_lcm = AutoPipelineForText2Image.from_pretrained(model_lcm_id, torch_dtype=torch.float16, variant="fp16", safety_checker = None, requires_safety_checker = False)
        pipe_lcm.scheduler = LCMScheduler.from_config(pipe_lcm.scheduler.config)
        pipe_lcm.to("cuda") # Use GPU for faster generation

        # load and fuse lcm lora
        pipe_lcm.load_lora_weights(adapter_lcm_id)
        pipe_lcm.fuse_lora()

        # Load YOLO model (using a pre-trained model, e.g., YOLOv5)
        model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s') #noa added 15.8.24

        prompt_num_object = self.count_objects_by_indices(prompt, object_indices=indices_to_alter) #noa added 15.8.24 - dict of prompt
        print(f'prompt_num_object is: {prompt_num_object}')

        # Initialize a list to store loss values - noa added 19.8.24
        loss_values = []

        pipe_line_type = 'new'
        #-------------------------noa added 14-15.8.24 - end

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()

                    #---------------------------------------------------------------------------------------------noa added 14.8.24 - for LCM model - start 
                    if pipe_line_type == 'old': #if running old pipeline simulatnusly...             
                        # Generate the image using the pipeline and latent variables
                        image_lcm = pipe_lcm(
                            prompt=prompt,
                            #prompt_embeds=prompt_embeds,
                            num_inference_steps=8,
                            generator=generator,
                            guidance_scale=2.0,
                            latent_vars=latents  # Pass the latent variables here
                        ).images[0]
                        # save the generated image
                        try:
                            Path.mkdir(f'./outputs/{prompt}')
                        except Exception as e:
                            print(e)
                            pass
                        #image_lcm.save(f'./outputs/{prompt}/lcm_denoise_step_{i}.png')
                        image_lcm.save(f'./outputs/lcm_denoise_step_{i}.png')

                        # Convert the PIL image to a format suitable for YOLO (numpy array)
                        image_lcm_np = np.array(image_lcm) #noa added 15.8.24
                        # Perform object detection on the generated image
                        results_yolo = model_yolo(image_lcm_np)#noa added 15.8.24
                        print(f'results_yolo in iter {i} are: {results_yolo}')#noa added 15.8.24

                        detector_num_object = self.object_dict_from_dtector(results_yolo) #noa added 15.8.24 - dict of detector  
                        print(f'detector_num_object is: {detector_num_object}')

                        prompt_num_object=self.count_objects_by_indices(prompt,object_indices=indices_to_alter) # ohad added 17.8
                        print(f'prompt_num_object is: {prompt_num_object}')

                        loss_lcm = self._compute_loss_make_it_count_project2(prompt_num_object_main=prompt_num_object, detector_num_object_main=detector_num_object) # noa added 15.8.24 (use Ohad's functions output)
                        print(f'loss_lcm in iter {i} are: {loss_lcm}')#noa added 15.8.24          

                        loss_lcm2 = self._compute_loss_with_tv(prompt_num_object_main=prompt_num_object, detector_num_object_main= detector_num_object, latents=latents)  
                        print(f'loss_lcm2 in iter {i} are: {loss_lcm2}')#noa added 15.8.24                                                                                                                         
                        #----------------------------------------------------------------------------------------------noa added 14.8.24 - for LCM model - end

                        # Get max activation value for each subject token
                        max_attention_per_index = self._aggregate_and_get_max_attention_per_token( #remove - we do not need in our project - noa - 15.8.24
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=sd_2_1)

                        if not run_standard_sd: #I do not think we need that *if* - Noa 15.8.24 (it is if we want to run the stable diffusion without attend-and-excite changes)
                            loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
 
                            # If this is an iterative refinement step, verify we have reached the desired threshold for all
                            if i in thresholds.keys() and loss > 1. - thresholds[i]:
                                del noise_pred_text
                                torch.cuda.empty_cache()
                                loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                    latents=latents,
                                    indices_to_alter=indices_to_alter,
                                    loss=loss,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_embeds,
                                    text_input=text_inputs,
                                    attention_store=attention_store,
                                    step_size=scale_factor * np.sqrt(scale_range[i]),
                                    t=t,
                                    attention_res=attention_res,
                                    smooth_attentions=smooth_attentions,
                                    sigma=sigma,
                                    kernel_size=kernel_size,
                                    normalize_eot=sd_2_1)

                            # Perform gradient update - #we should keep it (original)- noa 15.8.24 (think about the if statment)
                            if i < max_iter_to_alter:
                                loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                                # loss = self._compute_loss_make_it_count_project(prompt_num_object_main=prompt_num_object, detector_num_object_main=detector_num_object) #- noa added 15.8.24 (use Ohad's functions output)
                                if loss != 0:
                                    latents = self._update_latent(latents=latents, loss=loss,
                                                                step_size=scale_factor * np.sqrt(scale_range[i]))
                                print(f'Iteration {i} | Loss: {loss:0.4f}')

                    else: #noa added 18.8.24 - for our pipeline
                    #---------------------------------------------------------------------------------------------noa added 14.8.24 - for LCM model - start               
                        # Generate the image using the pipeline and latent variables
                        image_lcm = pipe_lcm(
                            prompt=prompt,
                            #prompt_embeds=prompt_embeds,
                            num_inference_steps=8,
                            generator=generator,
                            guidance_scale=3.0,
                            latent_vars=latents  # Pass the latent variables here
                        ).images[0]
                        # save the generated image
                        try:
                            Path.mkdir(f'./outputs/{prompt}')
                        except Exception as e:
                            print(e)
                            pass
                        #image_lcm.save(f'./outputs/{prompt}/lcm_denoise_step_{i}.png')
                        image_lcm.save(f'./outputs/lcm_denoise_step_{i}.png')

                        # Convert the PIL image to a format suitable for YOLO (numpy array)
                        image_lcm_np = np.array(image_lcm) #noa added 15.8.24
                        # Perform object detection on the generated image
                        results_yolo = model_yolo(image_lcm_np)#noa added 15.8.24
                        print(f'results_yolo in iter {i} are: {results_yolo}')#noa added 15.8.24

                        detector_num_object = self.object_dict_from_dtector(results_yolo) #noa added 15.8.24 - dict of detector  
                        print(f'detector_num_object is: {detector_num_object}')

                        #prompt_num_object=self.count_objects_by_indices(prompt,object_indices=indices_to_alter) # ohad added 17.8
                        print(f'prompt_num_object is: {prompt_num_object}')

                        #loss_lcm = self._compute_loss_make_it_count_project2(prompt_num_object_main=prompt_num_object, detector_num_object_main=detector_num_object) # noa added 15.8.24 (use Ohad's functions output)
                        #print(f'loss_lcm in iter {i} are: {loss_lcm}')#noa added 15.8.24      
                        
                        #loss = self._compute_loss_make_it_count_project2(prompt_num_object_main=prompt_num_object, detector_num_object_main=detector_num_object)# noa added 15.8.24 (use Ohad's functions output)
                        #print(f'loss_lcm in iter {i} are: {loss}')#noa added 15.8.24 
                         
                        loss = self._compute_loss_with_tv(prompt_num_object_main=prompt_num_object, detector_num_object_main= detector_num_object, latents=latents)  #noad added 18.8.24
                        #print(f'loss in iter {i} are: {loss}')#noa added 15.8.24  
                            
                        # Store the scalar value of the loss
                        loss_values.append(loss.item())  #noa added 19.8.24
                        
                        #Perform gradient update - Noa added 15.8.24 - Start
                        latents = self._update_latent(latents=latents, loss=loss,
                                                   step_size=scale_factor * np.sqrt(scale_range[i]))
                        print('latents were updated using lcm loss')
                        print(f'Iteration {i} | Loss: {loss:0.4f}')                                                                                                                                         
                        #----------------------------------------------------------------------------------------------noa added 14.8.24 - for LCM model - end

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance: 
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample #we should keep it (original)- noa 15.8.24

                # call the callback, if provided #we should keep it (original)- noa 15.8.24
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents) #we should keep it (original)- noa 15.8.24

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype) #think if we should keep it? - noa 15.8.24

        # 10. Convert to PIL #we should keep it (original)- noa 15.8.24 
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        #--------------------------Noa adeed 19.8.24 - START
        # Plotting the loss vs iteration
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs. Iteration')
        plt.legend()
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig('loss_vs_iteration.png')
    
        #--------------------------Noa added 19.8.24 - END

        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
