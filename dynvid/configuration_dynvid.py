"""
DynVidConfig: DynVid video token compression configuration.

Pipeline (post-projector):
  1. DySeg dynamic segmentation
  2. ERA group budget allocation (Participation Ratio)
  3. DATS anchor selection (Facility Location + dynamism-aware importance)
  4. Top-K Soft Fusion
  5. (optional) LLM internal pruning
"""


class DynVidConfig:
    """DynVid token compression configuration.

    Args:
        # -- Budget --
        before_LLM_retention_ratio (float): Token retention ratio after compression. Default 0.125.

        # -- DySeg Segmentation --
        dyseg_threshold (float): Cosine similarity threshold between adjacent frames;
            > threshold -> same segment. Default 0.4.
        min_segment_num (int): Minimum number of segments. Default 8.
        complementary_segment (bool): Auto-complement when fewer than min_segment_num. Default True.

        # -- ERA Group Budget Allocation --
        group_budget_method (str): "pr" | "uniform". Default "pr".

        # -- DATS Anchor Selection --
        anchor_method (str): "facility_location" | "topk". Default "facility_location".

        # -- Soft Fusion --
        topk_fusion (int): Number of anchors each dropped token is assigned to. Default 10.
        anchor_weight (float): fused = gamma*anchor + (1-gamma)*drop_centroid. Default 0.25.

        # -- LLM Internal Pruning --
        query_prune_layer (int): Pruning layer index, -1 = disabled. Default 20.
        llm_prune_ratio (float): Pruning ratio. Default 0.3.
    """

    VALID_GROUP_BUDGET_METHODS = ("uniform", "pr")
    VALID_ANCHOR_METHODS = ("facility_location", "topk")

    def __init__(
        self,
        # -- Budget --
        before_LLM_retention_ratio: float = 0.125,
        # -- DySeg --
        dyseg_threshold: float = 0.55,
        min_segment_num: int = 8,
        complementary_segment: bool = True,
        # -- ERA --
        group_budget_method: str = "pr",
        # -- DATS --
        anchor_method: str = "facility_location",
        # -- Soft Fusion --
        topk_fusion: int = 10,
        anchor_weight: float = 0.25,
        # -- LLM pruning --
        query_prune_layer: int = 20,
        llm_prune_ratio: float = 0.3,
    ):
        # Budget
        self.before_LLM_retention_ratio = before_LLM_retention_ratio

        # DySeg
        self.dyseg_threshold = dyseg_threshold
        self.min_segment_num = min_segment_num
        self.complementary_segment = complementary_segment

        # ERA
        assert group_budget_method in self.VALID_GROUP_BUDGET_METHODS
        self.group_budget_method = group_budget_method

        # DATS
        assert anchor_method in self.VALID_ANCHOR_METHODS
        self.anchor_method = anchor_method

        # Soft Fusion
        self.topk_fusion = topk_fusion
        assert 0.0 < anchor_weight <= 1.0
        self.anchor_weight = anchor_weight

        # LLM pruning
        self.query_prune_layer = query_prune_layer
        self.llm_prune_ratio = llm_prune_ratio

        # -- Runtime state (written internally by the pipeline) --
        self._visual_token_range = None
        self._target_budget = 0

    def __repr__(self):
        return (
            f"DynVidConfig(\n"
            f"  before_LLM_retention_ratio={self.before_LLM_retention_ratio:.4f},\n"
            f"  dyseg_threshold={self.dyseg_threshold}, min_segment_num={self.min_segment_num},\n"
            f"  group_budget_method='{self.group_budget_method}',\n"
            f"  anchor_method='{self.anchor_method}',\n"
            f"  topk_fusion={self.topk_fusion}, anchor_weight={self.anchor_weight},\n"
            f"  query_prune_layer={self.query_prune_layer}, llm_prune_ratio={self.llm_prune_ratio},\n"
            f")"
        )