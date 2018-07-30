import os
import numpy as np
import trimesh
import time
from utils import (
    LOG_FORMAT,
    ProcessBase,
    basename_no_extension,
    save_trimesh_as_obj,
    stl_to_obj,
    trimesh_load_clean
)

class ProcessTightlyArranged(ProcessBase):
    def __init__(self):
        super().__init__("TightlyArranged")

    def perform(self, mesh_path, tmp_path, P2Slice_path):
        super().perform()

        my_mesh = trimesh_load_clean(mesh_path)
        meshes = my_mesh.split(only_watertight=False)

        number_of_meshes = len(meshes)

        if number_of_meshes <= 1:
            #  raise ValueError("No need for checking tightly arranged")
            return [mesh_path]

        sorted_meshes = sorted(meshes, key=lambda x:x.vertices[:,2].min())

        res = self.bullet_tightly_arranged_test(mesh_path, sorted_meshes, tmp_path, P2Slice_path) # make it self contained?
        if res is not None:
            self.logger.debug("{} is tightly arranged".format(mesh_path))
            return res

        return [mesh_path]


    def bullet_tightly_arranged_test(self, mesh_path, meshes,tmp_path, P2Slice_path):

        basename = basename_no_extension(mesh_path)
        basename_tmp_foler = os.path.join(tmp_path, basename)

        # TODO: handle this outside
        if len(meshes) < 2:
            return None

        if not os.path.exists(basename_tmp_foler):
            os.mkdir(basename_tmp_foler)

        tmp_stlnames = []
        tmp_objnames = []
        stlnames = []
        for counter, this_mesh in enumerate(meshes):
            name =  "{}_ta_{}".format(basename, counter) + ".{}"
            tmp_stlname = os.path.join(basename_tmp_foler, name.format("stl"))
            tmp_objname = os.path.join(basename_tmp_foler, name.format("obj"))

            # self.logger.debug(tmp_stlname)
            # self.logger.debug(tmp_objname)

            stlname = os.path.join(tmp_path, name.format("stl"))

            this_mesh.export(tmp_stlname)

            save_trimesh_as_obj(this_mesh, tmp_objname)

            tmp_stlnames.append(tmp_stlname)
            tmp_objnames.append(tmp_objname)
            stlnames.append(stlname)

        # bullet exec should exposed in dockerfile
        command = "bullet {} &> /dev/null".format(" ".join(tmp_objnames))

        self.logger.debug(command)
        start = time.time()
        command_out = os.system(command)
        end = time.time()
        #print("OUTPUT : ", command_out)
        #print("TEMPS POUR LA COMMANDE : ", end-start)
        for tmp_obj in tmp_objnames:
            os.remove(tmp_obj)


        self.logger.debug("output of bullet physics {}".format(command_out))

        bool_is_multipart = False
        if command_out == 0: # is multipart need to split everything
            bool_is_multipart = True

        if bool_is_multipart:
            self.logger.debug("is tightly arranged")
            for tmp_stl, stl in zip(tmp_stlnames, stlnames):
                # self.logger.debug("rename {} to {}".format(tmp_stl, stl))
                os.rename(tmp_stl, stl)
            os.rmdir(basename_tmp_foler)
            os.remove(mesh_path)
            return stlnames
        else:
            self.logger.debug("not tightly arranged")
            for tmp_stl in tmp_stlnames:
                os.remove(tmp_stl)
            os.rmdir(basename_tmp_foler)
            return None







