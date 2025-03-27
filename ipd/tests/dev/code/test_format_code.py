from evn.tests.test_detect_formatted_blocks import *
from evn.tests.test_filter_python_output import *
from evn.tests.test_formatter import *
from evn.tests.test_token_column_format import *

import ipd

def main():
    ipd.tests.maintest(globals())

if __name__ == '__main__':
    main()
