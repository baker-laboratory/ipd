import ipd

class SymTool(ipd.tools.IPDTool):
    ...

class BuildTool(SymTool):

    def from_components(self, comps: list[str]):
        print(comps)
        assert 0, 'not implemented'
