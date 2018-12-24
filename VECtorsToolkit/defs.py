import os

# Refactoring of the code https://github.com/gift-surg/lie_exponential


root_dir = os.path.abspath(os.path.dirname(__file__))
default_saver_loader_modality = 'txt'  # 'txt', 'npy' (or 'np')
default_image_saver_modality = 'nifti1'  # 'nifti1' or 'nifti2'

info = {
        "name": "VECtorsToolkit",
        "version": "0.0.0",
        "description": "Methods to manage vector fields.",
        "web_infos" : "",
        "repository": {
                       "type": "git",
                       "url": "https://github.com/SebastianoF/VECtorsToolkit"
                        },
        "author": "Sebastiano Ferraris",
        "dependencies": {
                        # requirements.txt file automatically generated using pipreqs.
                        "python" : "{0}/requirements.txt".format(root_dir)
                        # pip install -r requirements.txt to run the code
                        }
        }

vector_field_definition = \
"""
DEFINITION:\n Vector Field -> instance of the class np.array with 5 dimensions.
shape (x, y, z, t, d), sampling on a squared grid of the function

"""

omega_definition = \
"""
Omega provides the dimensions of the matrix where the vector field is sampled:
$v: Omega rightarrow mathbf{R}^d$
can be a tuple, a list or a 1d numpy array of len 2 or 3 of integers.
"""

v_shape_definition = \
"""
v_shape is the shape of the vector field in a numpy array sense.
\tIf Omega = (30,50), v_shape must be (30,50,1,1,2) \n
\tIf Omega = (30,40,50), v_shape must be (30,40,50,1,3)\n
\tIf Omega = (30,40,50) and time points t=10, v_shape must be (30,40,50,10,3)\n
"""

eulerian_lagrangian_definitions = \
"""
DEFINITIONS:
Eulerian coordinates -> absolute external coordinate frame (also deformation field or position field)
Lagrangian coordinates -> coordinate frame provided in respect to the particle (also displacement field)

As in ITK, vector fields are always considered both as input and output in Eulerian coordinates.
Fields are converted to Lagrangian coordinates only by methods that call a composition. And only internally.

In case the vector_field_manager is bypassed, remember that flexibility was preferred over safety:
the user NEEDS to know if he is dealing with a vector field in Lagrangian coordinates or one in Eulerian.
"""