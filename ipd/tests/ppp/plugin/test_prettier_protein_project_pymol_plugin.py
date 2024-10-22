import ipd
from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import *

at = ipd.lazyimport('assertpy', pip=True).assert_that

def main():
    run_pymol()
    # run_plugin_standalone()
    print('test_ppppp DONE', flush=True)

def run_polls_stress_test():
    server, backend = ipd.ppp.server.run(12345, 'postgresql://sheffler@192.168.0.154:5432/ppp')
    client = ipd.ppp.PPPClient('127.0.0.1:12345')
    polls = client.pollinfo()
    print(len(polls))
    print(polls[0])
    ipd.dev.global_timer.report()
    server.stop()

def run_plugin_standalone():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    run()
    print('^' * 80)

def run_pymol():
    # os.environ['QT_QPA_PLATFORM'] = 'xcb'
    pymol.pymol_argv = ['pymol', '-q']
    pymol.finish_launching()
    # time.sleep(1)
    # pymol.cmd.do('plugin_load ppppp')
    # ipd.ppp.plugin.ppppp.run()

if __name__ == '__main__':
    main()
