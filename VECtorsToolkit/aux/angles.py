import math


# ----Auxiliary function for transformations so and se ----


def mod_pipi(alpha):
    """
    mod_pipi(alpha) \n
    :param alpha: angle in rad
    :return: equivalent alpha in (-pi, pi]
    """
    if alpha > 2 * math.pi:
        alpha %= (2 * math.pi)
    elif alpha < -2 * math.pi:
        alpha %= (-2 * math.pi)

    if alpha > math.pi:
        alpha -= 2 * math.pi
    elif alpha <= -math.pi:
        alpha += 2 * math.pi

    return alpha
