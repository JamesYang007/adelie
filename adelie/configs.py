from .adelie_core import Configs


def set_configs(
    name: str,
    value =None,
):
    if value is None:
        value = getattr(Configs, name + "_def")
    setattr(Configs, name, value)
