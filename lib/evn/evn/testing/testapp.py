from evn import CLI
from pathlib import Path

# === Root CLI scaffold using inheritance-based hierarchy ===
class TestApp(CLI):
    """
    Main entry point for the EVN developer workflow CLI.
    """

    def version(self, verbose: bool = False):
        """
        Show version info.

        :param verbose: If True, include environment and git info.
        :type verbose: bool
        """
        print(f'[evn.version] Version info (verbose={verbose})')

    def name_with_under_scores(self):
        ...

    def _private_meth(self):
        ...

    @classmethod
    def _private_func(cls):
        ...

    @staticmethod
    def _static_func():
        ...

    @classmethod
    def _callback(cls, foo='bar'):
        """
        Configure global CLI behavior (used for testing _callback hooks).
        """
        return dict(help_option_names=['-h', '--help'])


class dev(TestApp):
    "Development: edit, format, test a single file or unit."

    pass


class format(dev):

    def stream(self, tab_width: int = 4, language: str = 'python'):
        """
        Format input stream.

        :param tab_width: Tab width to use.
        :type tab_width: int
        :param language: Programming language (e.g. 'python').
        :type language: str
        """
        print(
            f'[dev.format.stream] Format stream (tab_width={tab_width}, language={language})'
        )

    def smart(self, mode: str = 'git'):
        """
        Format changed project files.

        :param mode: Change detection mode ('md5', 'git').
        :type mode: str
        """
        print(f'[dev.format.smart] Format changed files using mode={mode}')


class test(dev):

    def file(self, fail_fast: bool = False):
        """
        Run pytest or doctest.

        :param fail_fast: Stop after first failure.
        :type fail_fast: bool
        """
        print(f'[dev.test.file] Run tests (fail_fast={fail_fast})')

    def swap(self, path: Path = Path('')):
        """
        Swap between test/source.

        :param path: Path to swap.
        :type path: Path
        """
        print(f'[dev.test.swap] Swap source/test for {path}')


class validate(dev):

    def file(self, strict: bool = True):
        """
        Validate file syntax/config.

        :param strict: Fail on warnings.
        :type strict: bool
        """
        print(f'[dev.validate.file] Validate file (strict={strict})')


class doc(dev):

    def build(self, open_browser: bool = False):
        """
        Build docs for current file.

        :param open_browser: Open result in browser.
        :type open_browser: bool
        """
        print(f'[dev.doc.build] Build docs (open_browser={open_browser})')


class create(dev):

    def testfile(self,
                 module: Path,
                 testfile: Path = Path(''),
                 prompts=True,
                 browser: str = ''):
        """
        Create a test file for current file.

        :param prompts: create prompts for ai gen.
        :type bool: bool
        """
        print(f'[dev.doc.build] Build docs (open_browser={browser})')


class doccheck(TestApp):
    "Doccheck: audit project documentation and doctests."

    @classmethod
    def _callback(cls, docsdir='docs'):
        return dict(help_option_names=['--dochelp'])


class build(doccheck):

    def full(self, force: bool = False):
        print(f'[doccheck.build.full] Full doc build (force={force})')


class open(doccheck):

    def file(self, browser: str = 'firefox'):
        print(f'[doccheck.open.file] Open HTML with browser={browser}')


class doctest(doccheck):

    def fail_loop(self, verbose: bool = False):
        print(
            f'[doccheck.doctest.fail_loop] Iterate doctest failures (verbose={verbose})'
        )


class missing(doccheck):

    def list(self, json: bool = False):
        print(f'[doccheck.missing.list] List missing docs (json={json})')


class qa(TestApp):
    "QA: prepare commits, PRs, and run test matrices."

    pass


class matrix(qa):

    def run(self, parallel: int = 1):
        print(f'[qa.matrix.run] Run matrix with {parallel} parallel jobs')


class testqa(qa):

    def loop(self, max_retries: int = 3):
        print(f'[qa.test.loop] Retry failing tests up to {max_retries} times')


class out(qa):

    def filter(self, min_lines: int = 5):
        print(f'[qa.out.filter] Filter output (min_lines={min_lines})')


class review(qa):

    def coverage(self, min_coverage: float = 75):
        print(f'[qa.review.coverage] Minimum coverage = {min_coverage}%')

    def changes(self, summary: bool = True):
        print(f'[qa.review.changes] Show changes (summary={summary})')


class run(TestApp):
    "Run: dispatch actions, scripts, or simulate GH actions."

    pass


class dispatch(run):

    def file(self, path: str):
        print(f'[run.dispatch.file] Dispatch on file {path}')


class act(run):

    def job(self, name: str):
        print(f'[run.act.job] Run GitHub job {name}')


class doit(run):

    def task(self, name: str = ''):
        print(f'[run.doit.task] Run doit task {name}')


class script(run):

    def shell(self, cmd: str):
        print(f'[run.script.shell] Run shell: {cmd}')


class buildtools(TestApp):
    "Build: C++ and native build tasks."

    pass


class cpp(buildtools):

    def compile(self, debug: bool = False):
        print(f'[build.cpp.compile] Compile (debug={debug})')

    def pybind(self, header_only: bool = False):
        print(
            f'[build.cpp.pybind] Generate pybind (header_only={header_only})')


class clean(buildtools):

    def all(self, verbose: bool = False):
        print(f'[build.clean.all] Clean all (verbose={verbose})')


class proj(TestApp):
    "Project structure, tagging, and discovery."

    def root(self, verbose: bool = False):
        print(f'[proj.TestApp] Project TestApp (verbose={verbose})')

    def info(self):
        print('[proj.info] Project metadata')

    def tags(self, rebuild: bool = False):
        print(f'[proj.tags] Generate tags (rebuild={rebuild})')


if __name__ == '__main__':
    # for click_path in TestApp._walk_click():
    # print(click_path)
    TestApp._run()
