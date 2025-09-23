class BodyManagerBase:
    """Manager for each body component (e.g., single arm, mobile base)."""

    def __init__(self, env, body_config):
        self.env = env
        self.body_config = body_config


class BodyConfigBase:
    """Configuration  for each body component (e.g., single arm, mobile base)."""

    pass
