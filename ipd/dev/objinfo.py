def info(
    obj,
    title=None,
    methods=True,
    docs=True,
    private=False,
    dunder=False,
    sort=True,
    all=False,
    value=True,
):
    import rich
    rich.inspect(
        obj,
        title=title,
        methods=methods,
        docs=docs,
        private=private,
        dunder=dunder,
        sort=sort,
        all=all,
        value=value,
    )
