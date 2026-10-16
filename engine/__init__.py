# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MatteDesk engine package."""
import os

# EXR support in OpenCV must be enabled before cv2 is imported anywhere in the engine
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# Unicode paths (e.g. "październik") – cv2.imread/imwrite on Windows fail silently without this
from engine.cv2_unicode import install as _install_cv2_unicode  # noqa: E402

_install_cv2_unicode()
