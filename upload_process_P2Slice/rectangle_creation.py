# %load /home/mmf159/code/cdn/utils/upload_process_P2Slice/rectangle_creation_optimised.py
import numpy as np
from stl import mesh
import time
import MinimumBoundingBox

#my_mesh = mesh.Mesh.from_file('stamp.stl')
"""
DATA Structure :
Rectangle R = [x_min, x_max, y_min, y_max]

What we want to get :
Object O = ['myfile.stl', R]
where the stl file is correctly oriented in the (xOy) plan and where R is the best rectangle that contains the object
Rectangle R = [x_min, x_max, y_min, y_max]

"""

def best_theta_new(my_mesh):
    # https://github.com/BebeSparkelSparkel/MinimumBoundingBox/blob/master/MinimumBoundingBox.py
    x_y = my_mesh.vectors[:,:,:2].reshape(-1, 2)
    result = MinimumBoundingBox.minimum_bounding_box(x_y)
    return result.unit_vector_angle%(2*np.pi)

def best_rectangle(namefile):
    import time
    """
    Searches for the best rotation to apply in the plan (xOy) to get the basic rectangle with the smaller surface area possible
    :param namefile: str: name of the stl file that has to be positioned on the bed
    :return: optimal surface area, coordinates of the rectangle, rotation to apply.
    """
    # calculates the best orientation for the object so that the rectangle is optimal
    """ old
    my_mesh = mesh.Mesh.from_file(namefile)
    best_theta = dicho_gradient_slope(f, gradient_f, my_mesh, 0)%(2*np.pi)
    """

    my_mesh = mesh.Mesh.from_file(namefile)
    best_theta = best_theta_new(my_mesh)

    #we rotate the file
    my_mesh.rotate(np.array([0, 0, 1]), best_theta)  # , point = np.array([0, 0, 0]))
    # updating all the data
    my_mesh.update_areas()
    my_mesh.update_max()
    my_mesh.update_min()
    my_mesh.update_normals()


    # save the modified mesh

    my_mesh.save(namefile)

    height = my_mesh.z.max() - my_mesh.z.min()
    param = [float(my_mesh.x.min()),
            float(my_mesh.x.max()),
            float(my_mesh.y.min()),
            float(my_mesh.y.max())]

    # object returns in the wanted shape. We don't need the surface area anymore
    return best_theta, {"name":namefile, "best_rectangle": {"x_min": param[0], "x_max": param[1], "y_min": param[2], "y_max": param[3]}}, height
