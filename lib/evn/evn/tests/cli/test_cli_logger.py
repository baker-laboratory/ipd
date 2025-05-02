from evn.cli.cli_logger import CliLogger


class DummyCLI:
    __log__ = []

    @classmethod
    def get_command_path(cls):
        return 'DummyCLI'


def test_basic_logging():
    CliLogger.clear(DummyCLI)
    CliLogger.log(DummyCLI,
                  'test message',
                  event='test_event',
                  data={'key': 123})
    log = CliLogger.get_log(DummyCLI)
    assert len(log) == 1
    entry = log[0]
    assert entry['message'] == 'test message'
    assert entry['event'] == 'test_event'
    assert entry['data'] == {'key': 123}
    assert 'timestamp' in entry
    assert entry['path'] == 'DummyCLI'


def test_log_clear():
    CliLogger.log(DummyCLI, 'will clear')
    assert CliLogger.get_log(DummyCLI)
    CliLogger.clear(DummyCLI)
    assert CliLogger.get_log(DummyCLI) == []


def test_log_event_context(capsys):
    CliLogger.clear(DummyCLI)
    with CliLogger.log_event_context(DummyCLI,
                                     'context_event',
                                     data={'foo': 'bar'}):
        pass
    log = CliLogger.get_log(DummyCLI)
    assert len(log) == 2
    assert log[0]['message'] == 'begin context_event'
    assert log[1]['message'] == 'end context_event'
    assert log[0]['data'] == log[1]['data'] == {'foo': 'bar'}


def test_class_logging_does_not_call_instance_method():

    class DummyWithInstancePath:
        __log__ = []

        @classmethod
        def get_command_path(cls):
            return 'DummyWithInstancePath'

        def get_full_path(self):
            return 'fail-if-called'

    # Call logger with class, not instance
    CliLogger.log(DummyWithInstancePath, 'hello')
    log = CliLogger.get_log(DummyWithInstancePath)
    assert log[0]['path'] == 'DummyWithInstancePath'
