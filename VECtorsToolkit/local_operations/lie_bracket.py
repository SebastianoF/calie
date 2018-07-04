
def lie_bracket():
    pass


# def lie_bracket(left, right):
#         """
#         Compute the Lie bracket of two velocity fields.
        
#         Parameters:
#         -----------
#         :param left: Left velocity field.
#         :param right: Right velocity field.
#         Order of Lie bracket: [left,right] = Jac(left)*right - Jac(right)*left
#         :return Return the resulting velocity field
#         """

#         left_jac = helper.compute_jacobian_field(left.field)
#         right_jac = helper.compute_jacobian_field(right.field)

#         result = SVF()
#         result.init_field(left.field)
#         num_dims = sum(np.array(result.field.vol_ext) > 1)

#         if num_dims == 2:
#             result.field.data[..., 0] = \
#                 (left_jac.data[..., 0] * right.field.data[..., 0] +
#                  left_jac.data[..., 1] * right.field.data[..., 1]) -\
#                 (right_jac.data[..., 0] * left.field.data[..., 0] +
#                  right_jac.data[..., 1] * left.field.data[..., 1])
              
#             result.field.data[..., 1] = \
#                 (left_jac.data[..., 2] * right.field.data[..., 0] +
#                  left_jac.data[..., 3] * right.field.data[..., 1]) - \
#                 (right_jac.data[..., 2] * left.field.data[..., 0] +
#                  right_jac.data[..., 3] * left.field.data[..., 1])
#         else:
#             result.field.data[..., 0] = \
#                 (left_jac.data[..., 0] * right.field.data[..., 0] +
#                  left_jac.data[..., 1] * right.field.data[..., 1] +
#                  left_jac.data[..., 2] * right.field.data[..., 2]) - \
#                 (right_jac.data[..., 0] * left.field.data[..., 0] +
#                  right_jac.data[..., 1] * left.field.data[..., 1] +
#                  right_jac.data[..., 2] * left.field.data[..., 2])
              
#             result.field.data[..., 1] = \
#                 (left_jac.data[..., 3] * right.field.data[..., 0] +
#                  left_jac.data[..., 4] * right.field.data[..., 1] +
#                  left_jac.data[..., 5] * right.field.data[..., 2]) - \
#                 (right_jac.data[..., 3] * left.field.data[..., 0] +
#                  right_jac.data[..., 4] * left.field.data[..., 1] +
#                  right_jac.data[..., 5] * left.field.data[..., 2])

#             result.field.data[..., 2] = \
#                 (left_jac.data[..., 6] * right.field.data[..., 0] +
#                  left_jac.data[..., 7] * right.field.data[..., 1] +
#                  left_jac.data[..., 8] * right.field.data[..., 2]) - \
#                 (right_jac.data[..., 6] * left.field.data[..., 0] +
#                  right_jac.data[..., 7] * left.field.data[..., 1] +
#                  right_jac.data[..., 8] * left.field.data[..., 2])

#         return result
