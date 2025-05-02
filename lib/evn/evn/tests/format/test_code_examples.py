import re
import evn

config_test = evn.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)


def helper_test_code_examples(testname, original, reference):
    formatted = evn.format.format_buffer(original)
    if formatted != reference:
        print(testname)
        evn.diff(formatted, reference)


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        chrono=False,
    )


def read_examples():
    examples = []
    with open(f'{evn.pkgroot}/tests/format/_code_examples.pytxt') as inp:
        re_testname = re.compile(r'#####\s*test\s+(.*?)\s*###########')
        group = []
        context = None
        for line in inp:
            if not line.startswith('#$@!'):
                group.append(line)
            elif name := re_testname.findall(line):
                assert len(name) == 1
                if examples and not examples[-1][2]:
                    examples[-1][2].extend(examples[-1][1])
                examples.append([name[0], [], []])
                group = examples[-1][1]
            elif '↑ original ↓ formatted' in line:
                group = examples[-1][2]
        for e in examples:
            e[1] = ''.join(e[1])
            e[2] = ''.join(e[2])
    return examples


evn.testing.generate_tests(read_examples())

if __name__ == '__main__':
    main()
