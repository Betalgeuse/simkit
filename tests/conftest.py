import os
import sys

os.environ.pop("SIMKIT_PARAMS", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simkit"))
