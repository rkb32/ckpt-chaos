"""python -m ckpt_chaos <run|bench|judge> ...   (same as the `ckpt-chaos` command)"""
import sys

from .cli import main

sys.exit(main())
