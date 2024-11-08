class CrudCli:
    def __init_subclass__(cls, Client, **kw):
        super().__init_subclass__(**kw)
        cls.__spec_models__ = Client.__spec_models__
        cls.__client_models__ = Client.__client_models__
        add_basic_client_model_methods(cls, Client)

def add_basic_client_model_methods(cls, Client):
    for kind, Model in cls.__client_models__.items():
        # print(Model)
        # print([k for k, v in Model.__dict__.items() if callable(v) and not k.startswith(('_', 'model_'))])
        pass
