# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""PlateMatte: precise selective mattes for live-action plates (scene-linear EXR, backlit passes, SAM3).
Results are written as full-frame premultiplied EXRs so Nuke only reads them into SD_Fuse.
"""
from .plate_io import PlateLoadEXR, PlateToDisplay, PlateSaveEXR
from .backlit import BacklitCoverage, MaskRegionLimit
from .select_color import MaskEdgeComponents, PlateForegroundHDR
from .layers import PlateLoadMaskSeq, MatteSubtractFront, MaskMaxMerge, MaskTemporalMedian, MaskTemporalMedianMC, MaskRecoverFromGuide
from .split import MatteSplitByRegions
from .sam3_soft import SAM3_VideoTrackSoft

NODE_CLASS_MAPPINGS = {
    "PlateLoadEXR": PlateLoadEXR,
    "PlateToDisplay": PlateToDisplay,
    "PlateSaveEXR": PlateSaveEXR,
    "BacklitCoverage": BacklitCoverage,
    "MaskRegionLimit": MaskRegionLimit,
    "MaskEdgeComponents": MaskEdgeComponents,
    "PlateForegroundHDR": PlateForegroundHDR,
    "PlateLoadMaskSeq": PlateLoadMaskSeq,
    "MatteSubtractFront": MatteSubtractFront,
    "MaskMaxMerge": MaskMaxMerge,
    "MatteSplitByRegions": MatteSplitByRegions,
    "MaskTemporalMedian": MaskTemporalMedian,
    "MaskTemporalMedianMC": MaskTemporalMedianMC,
    "MaskRecoverFromGuide": MaskRecoverFromGuide,
    "SAM3_VideoTrackSoft": SAM3_VideoTrackSoft,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PlateLoadEXR": "Plate Load EXR (scene-linear)",
    "PlateToDisplay": "Plate To Display (preview / SAM)",
    "PlateSaveEXR": "Plate Save EXR (full frame RGBA)",
    "BacklitCoverage": "Backlit Coverage Matte (X2)",
    "MaskRegionLimit": "Mask Region Limit (selective)",
    "MaskEdgeComponents": "Mask Edge Components (objects entering frame)",
    "PlateForegroundHDR": "Plate Foreground Colour HDR (decontaminate)",
    "PlateLoadMaskSeq": "Plate Load Mask Sequence (regions)",
    "MatteSubtractFront": "Matte Subtract Front (depth order)",
    "MaskMaxMerge": "Mask Max Merge (union)",
    "MatteSplitByRegions": "Matte Split By Regions (seamless, depth order)",
    "MaskTemporalMedian": "Mask Temporal Median (remove 1-2 frame glitches)",
    "MaskTemporalMedianMC": "Mask Temporal Median MC (rigid objects, camera-compensated)",
    "MaskRecoverFromGuide": "Mask Recover From Guide (clean-plate diff, connected parts only)",
    "SAM3_VideoTrackSoft": "SAM3 Video Track + soft masks (probabilities for core stabilisation)",
}
