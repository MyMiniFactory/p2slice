import logging
import os
import argparse

import trimesh
import numpy as np

from utils import LOG_FORMAT, ProcessBase, append_data_to_json, does_accept_part, trimesh_load_clean
import functools
import operator

SPLIT_POSTFIX = "_split"

def clean_mesh(mesh):
    mesh.fix_normals()
    return split_mesh

#  def does_two_meshes_intersect(mesh_0, mesh_1):
    #  bbox_0_bound = mesh_0.bounding_box.bounds
    #  bbox_1_bound = mesh_1.bounding_box.bounds

    #  if (bbox_0_bound[1][0] < bbox_1_bound[0][0] or
        #  bbox_0_bound[0][0] > bbox_1_bound[1][0] or
        #  bbox_0_bound[1][1] < bbox_1_bound[0][1] or
        #  bbox_0_bound[0][1] > bbox_1_bound[1][1] or
        #  bbox_0_bound[1][2] < bbox_1_bound[0][2] or
        #  bbox_0_bound[0][2] > bbox_1_bound[1][2]):
        #  return False
    #  else:
        #  return True

def does_two_meshes_intersect(mesh_0, mesh_1):
    threshold = 0.0001

    bbox_0_bound = mesh_0.bounding_box.bounds
    bbox_1_bound = mesh_1.bounding_box.bounds

    max_x_range = threshold*max(bbox_0_bound[1][0] - bbox_0_bound[0][0] , bbox_1_bound[1][0] - bbox_1_bound[0][0])
    max_y_range = threshold*max(bbox_0_bound[1][1] - bbox_0_bound[0][1] , bbox_1_bound[1][1] - bbox_1_bound[0][1])
    max_z_range = threshold*max(bbox_0_bound[1][2] - bbox_0_bound[0][2] , bbox_1_bound[1][2] - bbox_1_bound[0][2])

    if (bbox_0_bound[1][0] + max_x_range < bbox_1_bound[0][0] or
        bbox_0_bound[0][0] > bbox_1_bound[1][0] + max_x_range or
        bbox_0_bound[1][1] + max_y_range < bbox_1_bound[0][1] or
        bbox_0_bound[0][1] > bbox_1_bound[1][1] + max_y_range or
        bbox_0_bound[1][2] + max_z_range < bbox_1_bound[0][2] or
        bbox_0_bound[0][2] > bbox_1_bound[1][2] + max_z_range):
        return False
    else:
        return True



def does_two_meshes_intersect_bbo(mesh_0, mesh_1):
#     print("bbo")
    try:
        m0_bboriented = mesh_0.bounding_box_oriented
        m1_bboriented = mesh_1.bounding_box_oriented
    except:
        return does_two_meshes_intersect(mesh_0, mesh_1)
    else:
        try:
            res = m0_bboriented.intersection(m1_bboriented, engine="scad")
            intersect = True
        except:
            intersect = False

        return intersect

def connectivity_test(m):

    def connected_components(neighbors):
        seen = set()
        def component(node):
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= neighbors[node] - seen
                yield node
        for node in neighbors:
            if node not in seen:
                yield component(node)

    old_graph = {}
    for i in range(len(m)):
        old_graph[i] = []

    for i in range(len(m)):
        for j in range(i+1, len(m)):
            if m[i][j] == 1:
                old_graph[i].append((i, j))
                old_graph[j].append((j, i))

    new_graph = {node: set(each for edge in edges for each in edge) for node, edges in old_graph.items()}

    components = []
    for component in connected_components(new_graph):
        c = set(component)
        components.append([edge for edges in old_graph.values()
                                for edge in edges
                                if c.intersection(edge)])
    res = []
    parts_not_in_group = list(range(len(m)))
    for group in components:
        res.append([])
        for c in group:
            res[-1].append(c[0])
            res[-1].append(c[1])
            if c[0] in parts_not_in_group:
                parts_not_in_group.remove(c[0])
            if c[1] in parts_not_in_group:
                parts_not_in_group.remove(c[1])
        res[-1] = list(set(res[-1]))

    for i in parts_not_in_group:# by its own
        res.append([i])

    res = [i for i in res if i != []]

    return res

class ProcessSplit(ProcessBase):
    def __init__(self):
        super().__init__("Split")

    def perform(self, mesh_path, export_folder_path, mesh_name, metadata_json_path, union=False):
        super().perform()

        split_stl_postfix = SPLIT_POSTFIX + ".stl"

        my_mesh = trimesh_load_clean(mesh_path)
        meshes = my_mesh.split(only_watertight=False)

        # TODO: remove small mesh
        accepted_meshes = [i for i in meshes if does_accept_part(i, my_mesh)]

        # debug save
        # self.logger.debug("save to tiger.stl for debugging")
        # combined_mesh=functools.reduce(operator.add, accepted_meshes)
        # combined_mesh.export(os.path.join(export_folder_path, "tiger.stl"))
        # debug save

        number_of_accepted = len(accepted_meshes)
        self.logger.debug("{} accept mesh".format(number_of_accepted))

        if len(accepted_meshes) == 0:
            raise ValueError("0 accept mesh")
        elif len(accepted_meshes) > 500:
            raise ValueError("too much accept mesh !")
        else:
            pass

        connection = self.get_intersection_percentage(accepted_meshes)

        self.logger.debug("connection {}".format(connection))
        number_of_splited_parts = len(connection)

        P2Slice_path = os.path.join(export_folder_path, mesh_name + split_stl_postfix)
        self.logger.debug("mesh_name {}".format(mesh_name))

        export_mesh_pathes = []
        for counter, group in enumerate(connection):

            combined_mesh=functools.reduce(operator.add, [accepted_meshes[i] for i in group])

            if number_of_accepted == 1:
                stl_name ="P2Slice_{}.stl".format(mesh_name)
            else:
                stl_name ="P2Slice_{}_{}{}.stl".format(mesh_name, counter, SPLIT_POSTFIX)

            self._export_and_append(combined_mesh, export_folder_path, stl_name, export_mesh_pathes)

        # make sure the following is correct
        #  append_data_to_json({"number_of_splited_parts": number_of_splited_parts}, metadata_json_path)
        self.logger.debug("{} exported mesh".format(number_of_splited_parts))

        self.add_data("export_mesh_pathes", export_mesh_pathes)

        self.logger.debug(export_mesh_pathes)
        return export_mesh_pathes

    def _export_and_append(self, mesh, export_folder_path, stl_name, export_mesh_pathes):
        export_path = os.path.join(export_folder_path, stl_name)
        self.logger.debug("export to {}".format(str(export_path)))
        mesh.export(export_path)
        export_mesh_pathes.append(export_path)

    def _remove_extremely_small_mesh(self, split_mesh):
        raise NotImplementedError



    #  @staticmethod
    def get_intersection_percentage(self, meshes):

        m = np.zeros([len(meshes), len(meshes)])
        m_1 = np.zeros([len(meshes), len(meshes)])

        intersection_list = []
        for i in range(len(meshes)):
            for j in range(i+1, len(meshes)):

                if does_two_meshes_intersect(meshes[i], meshes[j]):
                    m_1[i][j] = 1

                    # is the following two lines in the right indentation
                    intersection_list.append(i)
                    intersection_list.append(j)

        connection = connectivity_test(m_1)

        return connection

    def is_well_arranged(self, meshes, mesh, export_mesh_path):

        # number based on 12 cm machine where 0.01*12 = 0.01 which is like one layer height
        # assuming your scale the object for small printer (like STARTT)
        ACCEPTED_BASE_PERCENTAGE = 0.01

        all_bboxes = np.array([i.bounds for i in meshes])
        xmins = all_bboxes[:,0,0]
        xmaxs = all_bboxes[:,1,0]

        ymins = all_bboxes[:,0,1]
        ymaxs = all_bboxes[:,1,1]

        zmins = all_bboxes[:,0,2]
        zmaxs = all_bboxes[:,1,2]

        x_height = xmaxs.max() - xmins.min()
        y_height = ymaxs.max() - ymins.min()
        z_height = zmaxs.max() - zmins.min()

        assert(x_height > 0)
        assert(y_height > 0)
        assert(z_height > 0)

        allclose_xmin = np.allclose(xmins, np.mean(xmins), atol=ACCEPTED_BASE_PERCENTAGE*x_height)
        allclose_xmax = np.allclose(xmaxs, np.mean(xmaxs), atol=ACCEPTED_BASE_PERCENTAGE*x_height)
        allclose_ymin = np.allclose(ymins, np.mean(ymins), atol=ACCEPTED_BASE_PERCENTAGE*y_height)
        allclose_ymax = np.allclose(ymaxs, np.mean(ymaxs), atol=ACCEPTED_BASE_PERCENTAGE*y_height)
        allclose_zmin = np.allclose(zmins, np.mean(zmins), atol=ACCEPTED_BASE_PERCENTAGE*z_height)
        allclose_zmax = np.allclose(zmaxs, np.mean(zmaxs), atol=ACCEPTED_BASE_PERCENTAGE*z_height)

        if allclose_xmin or \
            allclose_xmax or \
            allclose_ymin or \
            allclose_ymax or \
            allclose_zmin or \
            allclose_zmax:
            self.logger.debug(
                "{}".format([allclose_xmin, allclose_xmax, allclose_ymin, allclose_ymax, allclose_zmin, allclose_zmax])
            )
            if allclose_zmin:
                pass
            elif allclose_zmax:
                #  raise NotImplementedError
                pass
            elif allclose_xmin:
                mesh.apply_transform(
                    np.array([1,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,1]).reshape(4, 4)
                )
                self.logger.info("allclose_xmin")
            elif allclose_xmax:
                mesh.apply_transform(
                    np.array([1,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,1]).reshape(4, 4)
                )
                self.logger.info("allclose_xmin")
                #  raise NotImplementedError
                pass
            elif allclose_ymin:
                #  raise NotImplementedError
                pass
            elif allclose_ymax:
                #  raise NotImplementedError
                pass
            else:
                #  raise NotImplementedError
                pass

            #  mesh.export(export_mesh_path)
            #  self.logger.info("save by well arranged")

            return True
        else:
            return False



if __name__ == "__main__":
    import sys
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    parser.add_argument(
        '--mesh_path',
        help="path to your mesh",
    )

    args = parser.parse_args()
    mesh_path = args.mesh_path
    my_mesh = trimesh_load_clean(mesh_path)
    print(split(my_mesh, os.path.basename(mesh_path)[:-4], os.path.dirname(mesh_path) + "/export"))
