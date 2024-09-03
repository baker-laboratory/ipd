import os
import ipd

_rms = ipd.cuda.build_extension("_rms", [
    os.path.abspath(f"{os.path.dirname(__file__)}/_qcp_rms.cpp"),
    os.path.abspath(f"{os.path.dirname(__file__)}/_qcp_rms.cu"),
], ['fit'], globals())

