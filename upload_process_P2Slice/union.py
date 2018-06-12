import trimesh
import numpy as np
import scipy

from utils import LOG_FORMAT, ProcessBase, append_data_to_json, does_accept_part, vertices_split, trimesh_load_clean

class ProcessUnion(ProcessBase):
    def __init__(self):
        super().__init__("Union")

    def perform(self, mesh_path, metadata_json_path):
        super().perform()

        my_mesh = trimesh_load_clean(mesh_path)
        meshes = my_mesh.split(only_watertight=False)

        number_of_meshes = len(meshes)

        if number_of_meshes <= 1:
            #  raise ValueError("No need for union")
            return

        self.logger.debug("trimesh split length {}".format(number_of_meshes))
        # self.logger.debug("tiger vertices split {}".format(vertices_split(my_mesh)))

        if not all([i.is_watertight for i in meshes]):
            return # dont do union since not all parts are watertight

        self.logger.debug("start doing union, all watertight")

        union_mesh = meshes[0]
        for i in meshes[1:]:
            try:
                union_mesh = union_mesh.union(i, engine="blender")
            except:
                raise ValueError("union command line call error")

        is_same_bounding_box = np.all(
            my_mesh.bounding_box.triangles == union_mesh.bounding_box.triangles)

        union_res = union_mesh.split(only_watertight=False)

        length_of_vertices_split = vertices_split(union_mesh, _type="mesh")
        if is_same_bounding_box and length_of_vertices_split == 1:
            self.logger.debug(
                "!Successful union! same bbox {} result length {} original length {} is_watertight {}".format(
                is_same_bounding_box,
                len(union_res),
                number_of_meshes,
                union_mesh.is_watertight
                )
            )
            #  combined_mesh = union_mesh
            #  union_res[0].export(mesh_path)
            union_mesh.export(mesh_path)
            union_success = True
        else:
            self.logger.debug("union not successfuly")
            self.logger.debug("same bbox {}".format(is_same_bounding_box))
            self.logger.debug(
                "result of union has vertices shell counter {}".format(length_of_vertices_split)
            )
            union_success = False

        append_data_to_json({"union_success": union_success}, metadata_json_path)

    def union_res_does_accept_part(self, split_mesh, original_mesh):
        area_percentage = split_mesh.area/original_mesh.area
        area_condition = area_percentage > 0.001

        if area_condition:
            return True
        else:
            return False
