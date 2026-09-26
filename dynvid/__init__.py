"""
__init__.py

DynVid: Dynamism-Aware Adaptive Video Token Compression

Pipeline (post-projector):
  Vision Encoder (unchanged) -> Projector
    -> DySeg dynamic segmentation (cosine similarity threshold)
    -> ERA adaptive budget allocation (Participation Ratio)
    -> DATS anchor selection (Facility Location + dynamism-aware importance)
    -> Top-K Soft Fusion (anchor-protected mean)
    -> LLM
    -> (optional) attention-based internal pruning

Usage (Qwen2.5-VL):
    from dynvid import dynvid_qwen2_5
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
    dynvid_qwen2_5(model, before_LLM_retention_ratio=0.125)

Usage (Qwen3-VL):
    from dynvid import dynvid_qwen3
    model = Qwen3VLForConditionalGeneration.from_pretrained(...)
    dynvid_qwen3(model, before_LLM_retention_ratio=0.125)

Usage (LLaVA-OneVision / LLaVA-Video):
    from dynvid import dynvid_llava
    model = load_pretrained_model(...)
    dynvid_llava(model, before_LLM_retention_ratio=0.125)
"""

from .configuration_dynvid import DynVidConfig

# ============================================================================
# Qwen2.5-VL imports
# ============================================================================
from .modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention_forward,
    Qwen2_5_VLForConditionalGeneration_generate,
    Qwen2_5_VLModel_forward,
    Qwen2_5_VLModel_get_video_features,
    Qwen2_5_VLTextModel_forward,
    Qwen2_5_VLVisionAttention_forward,
    Qwen2_5_VLVisionBlock_forward,
    Qwen2_5_VisionTransformerPretrainedModel_forward,
)

# ============================================================================
# Qwen3-VL imports (lazy)
# ============================================================================
_qwen3_vl_imported = False
_qwen3_vl_funcs = {}


def _ensure_qwen3_vl_imports():
    global _qwen3_vl_imported, _qwen3_vl_funcs
    if _qwen3_vl_imported:
        return
    from .modeling_qwen3_vl import (
        Qwen3VLVisionAttention_forward,
        Qwen3VLVisionBlock_forward,
        Qwen3VLVisionModel_forward,
        Qwen3VLModel_forward,
        Qwen3VLModel_get_video_features,
        Qwen3VLModel_get_image_features,
        Qwen3VLTextAttention_forward,
        Qwen3VLTextDecoderLayer_forward,
        Qwen3VLTextModel_forward,
        Qwen3VLForConditionalGeneration_generate,
    )
    _qwen3_vl_funcs.update(dict(
        Qwen3VLVisionAttention_forward=Qwen3VLVisionAttention_forward,
        Qwen3VLVisionBlock_forward=Qwen3VLVisionBlock_forward,
        Qwen3VLVisionModel_forward=Qwen3VLVisionModel_forward,
        Qwen3VLModel_forward=Qwen3VLModel_forward,
        Qwen3VLModel_get_video_features=Qwen3VLModel_get_video_features,
        Qwen3VLModel_get_image_features=Qwen3VLModel_get_image_features,
        Qwen3VLTextAttention_forward=Qwen3VLTextAttention_forward,
        Qwen3VLTextDecoderLayer_forward=Qwen3VLTextDecoderLayer_forward,
        Qwen3VLTextModel_forward=Qwen3VLTextModel_forward,
        Qwen3VLForConditionalGeneration_generate=Qwen3VLForConditionalGeneration_generate,
    ))
    _qwen3_vl_imported = True


# ============================================================================
# LLaVA imports (lazy)
# ============================================================================
_llava_imported = False
_llava_funcs = {}


def _ensure_llava_imports():
    global _llava_imported, _llava_funcs
    if _llava_imported:
        return
    from .modeling_llava_onevision import (
        SigLipAttention_forward,
        SigLipVisionTower_forward,
        LlavaMetaForCausalLM_encode_images,
        LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
        Qwen2Attention_forward,
        Qwen2DecoderLayer_forward,
        Qwen2Model_forward,
    )
    _llava_funcs.update(dict(
        SigLipAttention_forward=SigLipAttention_forward,
        SigLipVisionTower_forward=SigLipVisionTower_forward,
        LlavaMetaForCausalLM_encode_images=LlavaMetaForCausalLM_encode_images,
        LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal=LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
        Qwen2Attention_forward=Qwen2Attention_forward,
        Qwen2DecoderLayer_forward=Qwen2DecoderLayer_forward,
        Qwen2Model_forward=Qwen2Model_forward,
    ))
    _llava_imported = True


# ============================================================================
# Shared config builder
# ============================================================================
_TD_PARAM_DEFAULTS = dict(
    before_LLM_retention_ratio=0.125,
    dyseg_threshold=0.55,
    min_segment_num=8,
    complementary_segment=True,
    group_budget_method="pr",
    anchor_method="facility_location",
    topk_fusion=10,
    anchor_weight=0.25,
    query_prune_layer=20,
    llm_prune_ratio=0.3,
)


def _build_td_config(**kwargs):
    """Build DynVidConfig from kwargs, filling defaults from _TD_PARAM_DEFAULTS."""
    params = {k: kwargs.get(k, v) for k, v in _TD_PARAM_DEFAULTS.items()}
    return DynVidConfig(**params)


# ============================================================================
# Qwen2.5-VL entry point
# ============================================================================
def dynvid_qwen2_5(
    model,
    before_LLM_retention_ratio: float = 0.125,
    dyseg_threshold: float = 0.55,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    group_budget_method: str = "pr",
    anchor_method: str = "facility_location",
    topk_fusion: int = 10,
    anchor_weight: float = 0.25,
    query_prune_layer: int = 20,
    llm_prune_ratio: float = 0.3,
):
    """Register DynVid compression on Qwen2.5-VL."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLAttention,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLModel,
        Qwen2_5_VLTextModel,
        Qwen2_5_VLVisionAttention,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    assert isinstance(model, Qwen2_5_VLForConditionalGeneration)

    td_config = _build_td_config(**{k: v for k, v in locals().items() if k != 'model'})

    Qwen2_5_VLAttention.forward = Qwen2_5_VLAttention_forward
    Qwen2_5_VLModel.get_video_features = Qwen2_5_VLModel_get_video_features
    Qwen2_5_VLTextModel.forward = Qwen2_5_VLTextModel_forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_forward
    Qwen2_5_VLVisionAttention.forward = Qwen2_5_VLVisionAttention_forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = (
        Qwen2_5_VisionTransformerPretrainedModel_forward
    )
    Qwen2_5_VLForConditionalGeneration.generate_ori = (
        Qwen2_5_VLForConditionalGeneration.generate
    )
    Qwen2_5_VLForConditionalGeneration.generate = (
        Qwen2_5_VLForConditionalGeneration_generate
    )

    setattr(model, 'td_config', td_config)
    setattr(model.model, 'td_config', td_config)
    setattr(model.model.language_model, 'td_config', td_config)
    setattr(model.model.visual, 'td_config', td_config)

    print(f"[DynVid] Registered on Qwen2.5-VL:\n  {td_config}")
    return model


# ============================================================================
# Qwen3-VL entry point
# ============================================================================
def dynvid_qwen3(
    model,
    before_LLM_retention_ratio: float = 0.125,
    dyseg_threshold: float = 0.55,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    group_budget_method: str = "pr",
    anchor_method: str = "facility_location",
    topk_fusion: int = 10,
    anchor_weight: float = 0.25,
    query_prune_layer: int = 20,
    llm_prune_ratio: float = 0.3,
):
    """Register DynVid compression on Qwen3-VL."""
    _ensure_qwen3_vl_imports()

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLVisionAttention,
        Qwen3VLVisionBlock,
        Qwen3VLVisionModel,
        Qwen3VLModel,
        Qwen3VLTextAttention,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextModel,
    )

    assert isinstance(model, Qwen3VLForConditionalGeneration)

    td_config = _build_td_config(**{k: v for k, v in locals().items() if k != 'model'})

    f = _qwen3_vl_funcs
    Qwen3VLVisionAttention.forward = f['Qwen3VLVisionAttention_forward']
    Qwen3VLVisionBlock.forward = f['Qwen3VLVisionBlock_forward']
    Qwen3VLVisionModel.forward = f['Qwen3VLVisionModel_forward']
    Qwen3VLModel.forward = f['Qwen3VLModel_forward']
    Qwen3VLModel.get_video_features = f['Qwen3VLModel_get_video_features']
    Qwen3VLModel.get_image_features = f['Qwen3VLModel_get_image_features']
    Qwen3VLTextAttention.forward = f['Qwen3VLTextAttention_forward']
    Qwen3VLTextDecoderLayer.forward = f['Qwen3VLTextDecoderLayer_forward']
    Qwen3VLTextModel.forward = f['Qwen3VLTextModel_forward']
    Qwen3VLForConditionalGeneration.generate_ori = (
        Qwen3VLForConditionalGeneration.generate
    )
    Qwen3VLForConditionalGeneration.generate = (
        f['Qwen3VLForConditionalGeneration_generate']
    )

    setattr(model, 'td_config', td_config)
    setattr(model.model, 'td_config', td_config)
    setattr(model.model.language_model, 'td_config', td_config)
    setattr(model.model.visual, 'td_config', td_config)

    print(f"[DynVid] Registered on Qwen3-VL:\n  {td_config}")
    return model


# ============================================================================
# LLaVA-OneVision / LLaVA-Video entry point
# ============================================================================
def dynvid_llava(
    model,
    before_LLM_retention_ratio: float = 0.125,
    dyseg_threshold: float = 0.55,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    group_budget_method: str = "pr",
    anchor_method: str = "facility_location",
    topk_fusion: int = 10,
    anchor_weight: float = 0.25,
    query_prune_layer: int = 20,
    llm_prune_ratio: float = 0.3,
):
    """Register DynVid compression on LLaVA-OneVision / LLaVA-Video."""
    _ensure_llava_imports()

    try:
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    except ImportError as e:
        raise ImportError("LLaVA is not installed.") from e

    try:
        from llava.model.llava_arch import LlavaMetaForCausalLM
    except ImportError as e:
        raise ImportError("Cannot import LlavaMetaForCausalLM.") from e

    from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2DecoderLayer,
        Qwen2Model,
    )

    assert isinstance(model, LlavaQwenForCausalLM)

    td_config = _build_td_config(**{k: v for k, v in locals().items() if k != 'model'})

    f = _llava_funcs
    SigLipAttention.forward = f['SigLipAttention_forward']
    type(model.model.vision_tower).forward = f['SigLipVisionTower_forward']
    LlavaMetaForCausalLM.encode_images = f['LlavaMetaForCausalLM_encode_images']
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = f['LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal']
    Qwen2Attention.forward = f['Qwen2Attention_forward']
    Qwen2DecoderLayer.forward = f['Qwen2DecoderLayer_forward']
    Qwen2Model.forward = f['Qwen2Model_forward']

    setattr(model, 'td_config', td_config)
    setattr(model.model, 'td_config', td_config)

    print(f"[DynVid] Registered on LLaVA:\n  {td_config}")
    return model