import ipd

class CodeTool(ipd.tools.IPDTool):
    def make_testfile(self, sourcefile, testfile):
        ipd.dev.make_testfile(sourcefile, testfile)
