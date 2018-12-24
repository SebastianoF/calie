import numpy as np

from VECtorsToolkit.operations import jacobians as jac
from VECtorsToolkit.fields import queries as qr


def lie_bracket(left, right):
        """
        Compute the Lie bracket of two velocitys.
        
        Parameters:
        -----------
        :param left: Left velocity.
        :param right: Right velocity.
        Order of Lie bracket: [left,right] = Jac(left)*right - Jac(right)*left
        :return Return the resulting velocity
        """

        left_jac = jac.compute_jacobian(left)
        right_jac = jac.compute_jacobian(right)

        result = np.zeros_like(left)
        num_dims = qr.check_is_vf(result)

        if num_dims == 2:
            result[..., 0] = \
                (left_jac[..., 0] * right[..., 0] + left_jac[..., 1] * right[..., 1]) - \
                (right_jac[..., 0] * left[..., 0] + right_jac[..., 1] * left[..., 1])
              
            result[..., 1] = \
                (left_jac[..., 2] * right[..., 0] + left_jac[..., 3] * right[..., 1]) - \
                (right_jac[..., 2] * left[..., 0] + right_jac[..., 3] * left[..., 1])
        else:
            result[..., 0] = \
                (left_jac[..., 0] * right[..., 0] + left_jac[..., 1] * right[..., 1] + left_jac[..., 2] * right[..., 2]) - \
                (right_jac[..., 0] * left[..., 0] + right_jac[..., 1] * left[..., 1] + right_jac[..., 2] * left[..., 2])
              
            result[..., 1] = \
                (left_jac[..., 3] * right[..., 0] + left_jac[..., 4] * right[..., 1] + left_jac[..., 5] * right[..., 2]) - \
                (right_jac[..., 3] * left[..., 0] + right_jac[..., 4] * left[..., 1] + right_jac[..., 5] * left[..., 2])

            result[..., 2] = \
                (left_jac[..., 6] * right[..., 0] + left_jac[..., 7] * right[..., 1] + left_jac[..., 8] * right[..., 2]) - \
                (right_jac[..., 6] * left[..., 0] + right_jac[..., 7] * left[..., 1] + right_jac[..., 8] * left[..., 2])

        return result
