import ipd

def test_sym_options():
    opt = ipd.sym.get_sym_options()
    ipd.icv(opt)

if __name__ == '__main__':
    test_sym_options()
