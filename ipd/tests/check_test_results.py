import glob
import sys

def main():
    fail = False

    for f in glob.glob('pytest*.log'):
        with open(f) as inp:
            lines = inp.readlines()
            fail |= 'failed' in lines[-1]
            for line in lines:
                fail |= 'ERROR' in line
                fail |= 'FAILED' in line
                fail |= 'FATAL' in line
                fail |= 'Error while loading ' in line
        if fail:
            print('PYTEST FAILED:', f)
            print(str.join('', lines))

    with open('ruff.log') as inp:
        lines = inp.readlines()
        if 'All checks passed!' not in lines[-1]:
            print('RUFF FAILED')
            print(str.join('', lines))
            fail = True

    if fail:
        sys.exit(-1)

if __name__ == '__main__':
    main()
