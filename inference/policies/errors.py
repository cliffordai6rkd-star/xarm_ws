"""Checkpoint contract errors that the deployment CLI can explain directly."""


class JointActionContractError(ValueError):
    """The model interface or adapted target cannot produce valid joint commands."""
