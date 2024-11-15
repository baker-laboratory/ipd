import os

import yaml

import ipd

def isinstalled(installed, pkg):
    pkg = pkg.strip('"')
    if 'git+http' in pkg:
        pkg = pkg[:pkg.rfind('.git')]
        pkg = os.path.basename(pkg)
    else:
        for op in '>= <= == < > ='.split():
            pkg = pkg.split(op)[0]
    return pkg in installed

class MambaTool(ipd.tools.IPDTool):
    def install_yaml(self,
                     envfile,
                     secrets: str = '',
                     yes: bool = False,
                     overwrite: bool = False,
                     sequential: bool = False):
        self._add_secrets(secrets)
        with open(envfile) as inp:
            env = yaml.load(inp, yaml.CLoader)
        channels = ' '.join(f'-c {ch}' for ch in env['channels'] if ch != 'conda-forge')
        packages = [d for d in env['dependencies'] if not isinstance(d, str) or not d.startswith('python=')]
        # print(packages)
        pippackages = packages[packages.index('pip') + 1]['pip']  # type: ignore
        packages = packages[:packages.index('pip')]
        if not overwrite:
            installed = ipd.dev.run('mamba list | tail +4', echo=False).split(os.linesep)
            installed += ipd.dev.run('pip list | tail +3', echo=False).split(os.linesep)
            installed = {x.split()[0] for x in installed if x.strip()}
            packages = [x for x in packages if not isinstalled(installed, x)]
            pippackages = [x for x in pippackages if not isinstalled(installed, x)]
        yes = '-y' if yes else ''  # type: ignore
        if sequential:
            for p in packages:
                os.system(f'mamba install {yes} {channels} "{self._fill_secrets(p)}"')
            for p in pippackages:
                print(f'pip install "{self._fill_secrets(p)}"')
        else:
            packages = self._fill_secrets(' '.join(f'"{dep}"' for dep in packages))
            pippackages = self._fill_secrets(' '.join(f'"{dep}"' for dep in pippackages))
            if packages:
                mambacmd = f'mamba install {yes} {channels} {packages}'
                print(mambacmd)
                os.system(mambacmd)
            if pippackages:
                pipcmd = f'pip install {pippackages}'
                print(pipcmd)
                os.system(pipcmd)
