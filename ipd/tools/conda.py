import yaml
import os
import ipd

class MambaTool(ipd.tools.IPDTool):
    def install_yaml(self, envfile, secrets: str = '', yes: bool = False):
        self._add_secrets(secrets)
        with open(envfile) as inp:
            env = yaml.load(inp, yaml.CLoader)
        channels = ' '.join(f'-c {ch}' for ch in env['channels'] if ch != 'conda-forge')
        deps = [d for d in env['dependencies'] if not isinstance(d, str) or not d.startswith('python=')]
        # print(deps)
        pip = deps[deps.index('pip') + 1]['pip']
        deps = deps[:deps.index('pip')]
        deps = self._fill_secrets(' '.join(f'"{dep}"' for dep in deps))
        pipdeps = self._fill_secrets(' '.join(f'"{dep}"' for dep in pip))
        yes = '-y' if yes else ''
        mambacmd = f'mamba install {yes} {channels} {deps}'
        pipcmd = f'pip install {pipdeps}'
        os.system(mambacmd)
        os.system(pipcmd)
