from .adelie_core.configs import Configs


def set_configs(
    name: str,
    value =None,
):
    """Sets global configurations.

    See :class:`adelie.adelie_core.configs.Configs` to inspect the configuration variables.

    Parameters
    ----------
    name : str
        Configuration variable name.
    value : optional
        Value to assign to the configuration variable.
        If ``None``, the system default value is used.
        Default is ``None``.

    See Also
    --------
    adelie.adelie_core.configs.Configs
    """
    if value is None:
        value = getattr(Configs, name + "_def")
    setattr(Configs, name, value)
