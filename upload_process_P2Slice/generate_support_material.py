from stl import mesh
import numpy as np
import argparse
import sys
import os
import json

from slicing_config import SUPPORT_OVERHANG_ANGLE_RADIAN

# critical angle
# use the same number with the cura config overhange angle

"""
DATA STRUCTURE:

faces = [[x_0, y_0, z_0, x_1, y_1, z_1, x_2, y_2, z_2], ...]
list of the different faces of the mesh. Each sub-list represents a triangle of the mesh, and is made up of the coordinates of the 3 points that compose the triangle.

normals = [[x, y, z], ...]
list of the normals of the mesh's faces. normals[i] is the normal of faces[i]

list_min_x = [x_min_f0, x_min_f1, ...]
list_max_x = [x_max_f0, x_max_f1, ...]
"""


def sort_faces_able_to_support_support(my_mesh):
    """
    :param my_mesh: stl mesh.
    :return list_indexes: list of the indexes of the faces which normals are oriented 'towards the top' (the one that can indeed support support material)
            list_min_x: np.array: list of x-minimum for these faces
            list_max_x: np.array: list of x-maximum for these faces
            list_min_y: np.array: list of y-minimum for these faces
            list_max_y: np.array: list of y-maximum for these faces
            list_min_z: np.array: list of z-minimum for these faces
            list_max_z: np.array: list of z-maximum for these faces
            indexes_faces_support_needed: np.array: list of the indexes of the faces that need support material
    """
    list_indexes = np.arange(len(my_mesh.x))
    indexes_faces_support_needed = np.arange(len(my_mesh.x))

    norm_hor = np.sqrt(my_mesh.normals[:, 0] ** 2 + my_mesh.normals[:, 1] ** 2)
    # bool = (norm_hor != 0)
    with np.errstate(divide='ignore', invalid='ignore'): # ignore divide by 0 warning and invalid ingore for case when normals is 0, 0, 0
        theta = np.arctan(my_mesh.normals[:, 2]/norm_hor) # norm_hor = 0 is not error, ignore the warning

    not_nan_theta_bool = np.logical_not(np.isnan(theta))
    with np.errstate(invalid='ignore'): # theta can be nan happens when normal is 0, 0, 0 TODO: write a preprocess function to remove these triangles
        able_to_sup_bool = np.logical_and(theta>0, not_nan_theta_bool)

    list_indexes = list_indexes[able_to_sup_bool]
    # support overhang angle is the absolute angle with negative z axis
    '''
    ---------------------------------
                   /|\
                  / | \
                 /  |  \
                /   |   \
               /    |    \
              / soa | soa \

             negative z axis
    soa stands for support overhang angle
    '''
    with np.errstate(invalid='ignore'): # same with previous invalid ignore
        sup_bool = np.logical_and(theta <= -SUPPORT_OVERHANG_ANGLE_RADIAN + 0.01, not_nan_theta_bool)
    indexes_faces_support_needed = indexes_faces_support_needed[sup_bool]

    list_min_x = np.min(my_mesh.x[able_to_sup_bool], axis = 1)
    list_max_x = np.max(my_mesh.x[able_to_sup_bool], axis = 1)
    list_min_y = np.min(my_mesh.y[able_to_sup_bool], axis = 1)
    list_max_y = np.max(my_mesh.y[able_to_sup_bool], axis = 1)
    list_min_z = np.min(my_mesh.z[able_to_sup_bool], axis = 1)
    list_max_z = np.max(my_mesh.z[able_to_sup_bool], axis = 1)

    return list_indexes, list_min_x, list_max_x, list_min_y, list_max_y, list_min_z, list_max_z, indexes_faces_support_needed

def find_indexes_x2(list_min_x, list_max_x, x_min, x_max): # list_indexes,
    """
    Find the indexes of the faces which have a part on the segment [x_min, x_max]
    :param list_min_x: np.array: list of x-minimum of faces
    :param list_max_x: np.array: list of x-maximum of faces
    :param x_min: float > 0. Beginning of the segment we want to find the matching faces on
    :param x_max: float > 0. End of the segment we want to find the matching faces on
    :return: test3: boolean array: if applied to the list of indexes, will give the indexes of the faces that have an intersection on x-axis with the impu segment
    """

    test1 = list_min_x <= x_max
    test2 = list_max_x >= x_min
    test3 = np.logical_and(test1, test2)

    return test3



def find_index_highest_faces_below_faster(edge, list_min_x, list_max_x, list_min_y, list_max_y, list_min_z, list_max_z):
    """
    Find the index of nearest face below the edge "edge" in the list of available faces.
    :param edge: [p0, p1] (pi is an array 3 (a point)): segment between p0 and p1
    :param list_min_x: np.array: list of x-minimum of faces available to support support
    :param list_max_x: np.array: list of x-maximum of faces available to support support
    :param list_min_y: np.array: list of y-minimum of faces available to support support
    :param list_max_y: np.array: list of y-maximum of faces available to support support
    :param list_min_z: np.array: list of z-minimum of faces available to support support
    :param list_max_z: np.array: list of z-maximum of faces available to support support
    :return: highest_face_below_z_min: float: z_min-coordinate of the ihighest face below, on which the support must rely on.
    """
    # we only consider the segment of the face part of the outlining, and its projections on the faces below.
    p0 = edge[0]
    p1 = edge[1]

    x_min = min(p0[0], p1[0])
    x_max = max(p0[0], p1[0])
    y_min = min(p0[1], p1[1])
    y_max = max(p0[1], p1[1])
    z_min = min(p0[2], p1[2])

    bool_ok_x = find_indexes_x2(list_min_x, list_max_x, x_min, x_max)
    bool_ok_y_after_x = find_indexes_x2(list_min_y[bool_ok_x], list_max_y[bool_ok_x], y_min, y_max)

    bool_ok_z_after_y_after_x = list_max_z[bool_ok_x][bool_ok_y_after_x] < z_min

    if not bool_ok_z_after_y_after_x.any() :
        highest_face_below_z_min = 0
    else :
        highest_face_below_z_min = np.max(list_min_z[bool_ok_x][bool_ok_y_after_x][bool_ok_z_after_y_after_x])

    return highest_face_below_z_min

def find_outlining_hashing(indexes_support_needed, my_mesh):
    """
    Find the outlinings of support material
    :param indexes_support_needed: list of the indexes of the faces needing support
    :param my_mesh : stl mesh
    :return : list of edges that are part of the outlining
    """
    v0 = my_mesh.v0[indexes_support_needed]
    v1 = my_mesh.v1[indexes_support_needed]
    v2 = my_mesh.v2[indexes_support_needed]

    v0 = [tuple(i) for i in v0.tolist()]
    v1 = [tuple(i) for i in v1.tolist()]
    v2 = [tuple(i) for i in v2.tolist()]

    res = [tuple(sorted(list(i))) for i in zip(v0, v1)] \
          + [tuple(sorted(list(i))) for i in zip(v1, v2)] \
          + [tuple(sorted(list(i))) for i in zip(v0, v2)]

    # find the elements in res that only appear once

    d = {}
    for i in res: d[i] = i in d
    return [k for k in d if not d[k]]

def create_support_from_edges2(normals, list_min_x, list_max_x,
                               list_min_y, list_max_y, list_min_z, list_max_z, outlining_edges):
    """
    Create a stl file containing the outlining of support material that the input mesh need.
    """
    list_vertices = []
    list_faces = []
    edge_nb = 0

    for edge in outlining_edges:

        z_min_pillar = find_index_highest_faces_below_faster(edge,
                                                             list_min_x, list_max_x, list_min_y, list_max_y, list_min_z, list_max_z)

        if z_min_pillar < 0:
            z_min_pillar = 0

        p0 = edge[0]
        p0_proj = [p0[0], p0[1], z_min_pillar]
        p1 = edge[1]
        p1_proj = [p1[0], p1[1], z_min_pillar]

        list_vertices.append(p0)
        list_vertices.append(p0_proj)
        list_vertices.append(p1)
        list_vertices.append(p1_proj)
        list_faces.append([4 * edge_nb, 4 * edge_nb + 1, 4 * edge_nb + 3])
        list_faces.append([4 * edge_nb, 4 * edge_nb + 3, 4 * edge_nb + 2])

        edge_nb += 1

    supp_vertices = np.array(list_vertices)
    supp_faces = np.array(list_faces)
    # Create the mesh
    support_mesh = mesh.Mesh(np.zeros(supp_faces.shape[0], dtype=mesh.Mesh.dtype))
    n_vert = len(supp_vertices)
    n_faces = len(supp_faces)
    if n_vert != 0 and n_faces:
        for i, f in enumerate(supp_faces):
            for j in range(3):
                support_mesh.vectors[i][j] = supp_vertices[f[j], :]

        support_mesh.update_normals()
        support_mesh.update_areas()
        support_mesh.update_max()
        support_mesh.update_min()

    return support_mesh

def support_generation_process(mesh_path, output_support_stl_path):

    #  filename = os.path.split(mesh_path)[1].split('.')[0]

    my_mesh = mesh.Mesh.from_file(mesh_path)

    z_min = my_mesh.z.min()

    my_mesh.translate(np.array([0, 0, -z_min]))

    list_indexes, list_min_x2, list_max_x2, list_min_y, list_max_y, list_min_z, list_max_z, indexes_support_needed = sort_faces_able_to_support_support(
        my_mesh)
    outlining_edges = find_outlining_hashing(indexes_support_needed, my_mesh)

    mesh2 = create_support_from_edges2(my_mesh.normals, list_min_x2, list_max_x2, list_min_y,
                                                                 list_max_y, list_min_z, list_max_z, outlining_edges)
    mesh2.translate(np.array([0, 0, z_min]))

    #  P2Slice_dir_path = os.path.join(os.path.dirname(mesh_path), "P2Slice_tmp")
    #  json_file_name = os.path.join(P2Slice_dir_path, 'support_' + filename)

    #  support_json = '{}.json'.format(json_file_name)
    #  support_metadata_json = '{}_metadata.json'.format(json_file_name)

    #  upload_process_P2Slice.parsedata(mesh2,support_json,support_metadata_json)

    #  mesh2.save(os.path.join(output_dir_path, 'support_' + filename + '.stl'))
    mesh2.save(output_support_stl_path)
