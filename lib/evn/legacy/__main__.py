import argparse
import sys
import evn as evn


def get_args(sysargv):
    """get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        nargs='+',
                        default='',
                        help='use - for stdin')
    parser.add_argument('-f',
                        '--filter',
                        default='boilerplate',
                        choices=['', 'boilerplate'])
    parser.add_argument('-i', '--inplace', action='store_true')
    args = parser.parse_args(sysargv[1:])
    return args


def main():
    """Main function to execute the evn module."""
    args = get_args(sys.argv)
    for input_file in args.input:
        if input_file == '-':
            text = sys.stdin.read()
            args.inplace = False
        else:
            with open(input_file, 'r') as inp:
                text = inp.read()
        if args.filter:
            output = evn.filter_python_output(text, preset=args.filter)
        else:
            output = evn.format_buffer(text)
        ctx = open(input_file, 'w') if args.inplace else evn.just_stdout()
        with ctx as out:
            out.write(output)


if __name__ == '__main__':
    main()
