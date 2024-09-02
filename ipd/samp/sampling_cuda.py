import os
import ipd

ipd.cuda.build_extension("_sampling", [
    os.path.abspath(f"{os.path.dirname(__file__)}/_sampling.cpp"),
    os.path.abspath(f"{os.path.dirname(__file__)}/_sampling.cu"),
], ['samp'], globals())
