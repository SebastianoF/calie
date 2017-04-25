from math import pi


# ----Auxiliary function for transformations so and se ----


def mod_pipi(alpha):
    """
    mod_pipi(alpha) \n
    :param alpha: angle in rad
    :return: equivalent alpha in (-pi, pi]
    """
    if alpha > 2 * pi:
        alpha %= (2 * pi)
    elif alpha < -2 * pi:
        alpha %= (-2 * pi)

    if alpha > pi:
        alpha -= 2 * pi
    elif alpha <= -pi:
        alpha += 2 * pi

    return alpha
