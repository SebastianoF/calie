import os


def rigid_registration(pfi_tool,
                       pfi_input_fixed,
                       pfi_input_moving,
                       pfi_output_img,
                       pfi_output_matrix_transform,
                       open_mp_threads=4,
                       speed=False):

    pfi_reg_tool = os.path.join(pfi_tool, 'reg_aladin')
    command = pfi_reg_tool + \
              ' -ref ' + pfi_input_fixed + \
              ' -flo ' + pfi_input_moving + \
              ' -res ' + pfi_output_img + \
              ' -aff ' + pfi_output_matrix_transform + \
              ' -omp ' + str(open_mp_threads)

    output_msg = 'reg_aladin ' + \
                 ' -ref ' + os.path.basename(pfi_input_fixed) + \
                 ' -flo ' + os.path.basename(pfi_input_moving) + \
                 ' -res ' + os.path.basename(pfi_output_img) + \
                 ' -aff ' + os.path.basename(pfi_output_matrix_transform) + \
                 ' -omp ' + str(open_mp_threads)
    if speed:
        command += ' -lp 1 -speeeeed'
        output_msg += ' -lp 1 -speeeeed'

    return command, output_msg


def non_rigid_registration(pfi_tool,
                           pfi_input_fixed,
                           pfi_input_moving,
                           pfi_output_img,
                           pfi_output_control_point_grid,
                           open_mp_threads=4,
                           speed=False):

    pfi_reg_tool = os.path.join(pfi_tool, 'reg_f3d')
    command = pfi_reg_tool + \
              ' -ref ' + pfi_input_fixed + \
              ' -flo ' + pfi_input_moving + \
              ' -cpp ' + pfi_output_control_point_grid + \
              ' -res ' + pfi_output_img +  \
              ' -vel ' + \
              ' -omp ' + str(open_mp_threads)

    # cpp is a nii image
    output_msg = 'reg_f3d ' + \
                 ' -ref ' + os.path.basename(pfi_input_fixed) + \
                 ' -flo ' + os.path.basename(pfi_input_moving) + \
                 ' -cpp ' + os.path.basename(pfi_output_control_point_grid) + \
                 ' -res ' + os.path.basename(pfi_output_img) + \
                 ' -vel ' + \
                 ' -omp ' + str(open_mp_threads)

    if speed:
        command += ' -lp 3 -maxit 10'
        output_msg += ' -lp 3 -maxit 10'

    return command, output_msg


"""
Notes on how to OBTAIN SVFs from NiftyReg - wrapped with get_flow_field

SVF are obtained from NiftyReg registering A and B as follows.

1) reg_f3d with command -vel returns the corresponding cpp grid as the control point grid we are interested in.
2) The dense vector field that corresponds to the given gpp grid is then provided with -flow and it
is obtained in Eulerian coordinates.
3) To have it in Lagrangian coordinates for our elaboration, subtract the identity with python (and not with -disp 
   in niftyReg, otherwise it will be exponentiated again).
"""


def get_flow_field(pfi_tool,
                   pfi_input_fixed,
                   pfi_input_control_point_grid,
                   pfi_output_flow):
    
    pfi_reg_tool = os.path.join(pfi_tool, 'reg_transform')
    command = pfi_reg_tool + \
              ' -ref ' + pfi_input_fixed + \
              ' -flow ' + pfi_input_control_point_grid + \
              ' ' + pfi_output_flow

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(pfi_input_fixed) + \
                 ' -flow ' + os.path.basename(pfi_input_control_point_grid) + \
                 ' ' + os.path.basename(pfi_output_flow)

    return command, output_msg


def get_deformation_field(pfi_tool,
                          pfi_input_fixed,
                          pfi_input_control_point_grid,
                          pfi_output_def):

    pfi_reg_tool = os.path.join(pfi_tool, 'reg_transform')
    command = pfi_reg_tool + \
              ' -ref ' + pfi_input_fixed + \
              ' -def ' + pfi_input_control_point_grid + \
              ' ' + pfi_output_def

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(pfi_input_fixed) + \
                 ' -def ' + os.path.basename(pfi_input_control_point_grid) + \
                 ' ' + os.path.basename(pfi_output_def)

    return command, output_msg


def get_displacement_field(pfi_tool,
                           pfi_input_fixed,
                           pfi_input_control_point_grid,
                           pfi_output_disp):

    pfi_reg_tool = os.path.join(pfi_tool, 'reg_transform')
    command = pfi_reg_tool + \
              ' -ref ' + pfi_input_fixed + \
              ' -disp ' + pfi_input_control_point_grid + \
              ' ' + pfi_output_disp

    output_msg = 'reg_transform ' + \
                 ' -ref ' + os.path.basename(pfi_input_fixed) + \
                 ' -disp ' + os.path.basename(pfi_input_control_point_grid) + \
                 ' ' + os.path.basename(pfi_output_disp)

    return command, output_msg


def get_non_rigid_grid(pfi_tool,
                       pfi_input_fixed,
                       pfi_input_moving,
                       pfi_input_transformation,
                       pfi_output_grid):

    pfi_reg_tool = os.path.join(pfi_tool, 'reg_resample')
    command = pfi_reg_tool + \
              ' -ref ' + pfi_input_fixed + \
              ' -flo ' + pfi_input_moving + \
              ' -trans ' + pfi_input_transformation + \
              ' -blank ' + pfi_output_grid

    output_msg = 'reg_sample ' + \
                 ' -ref ' + os.path.basename(pfi_input_fixed) + \
                 ' -flo ' + os.path.basename(pfi_input_moving) + \
                 ' -trans ' + os.path.basename(pfi_input_transformation) + \
                 ' -blank ' + os.path.basename(pfi_output_grid)

    return command, output_msg
