def configure_penopt_schedule(schedule_type: str, c0: float, eta: float, update_c_every: int):
    """
    Configure QUAM penalty scheduling.

    Args:
        schedule_type (str): 'exp' (exponential), 'lin' (linear)
        c0 (float): initial value
        eta (float): growth factor (exp) or slope (lin)
        update_c_every (int): steps between updates

    Returns:
        Callable: function(step) -> float
    """
    if schedule_type == "exp":
        return lambda step: c0 * eta ** (step // update_c_every)
    elif schedule_type == "lin":
        return lambda step: c0 + eta * (step // update_c_every)
    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_type}")

        

