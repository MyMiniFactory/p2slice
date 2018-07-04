import os
import numpy as np
import trimesh
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

        is_well_arranged, transformed_mesh = self.is_well_arranged(my_mesh, meshes) # change the mymesh!

        if is_well_arranged:
            # the following line is for the
            # counter example https://www.myminifactory.com/object/1338
            # similar https://www.myminifactory.com/object/1405 object with support
            # hack since I know we fix the first mesh in the meshes so let the
            # splited mesh in meshes with biggest zmax to be on the first object
            split_t_meshes = transformed_mesh.split(only_watertight=False)
            sorted_meshes = sorted(split_t_meshes, key=lambda x:x.vertices[:,2].min())

            res = self.bullet_tightly_arranged_test(mesh_path, sorted_meshes, tmp_path, P2Slice_path) # make it self contained?
            if res is not None:
                self.logger.debug("{} is tightly arranged".format(mesh_path))
                return res

        return [mesh_path]


    def is_well_arranged(self, mesh, meshes):

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
                transformed_mesh = mesh
            elif allclose_zmax:
                transformed_mesh = mesh.apply_transform(
                        np.array([1,0,0,0,
                            0,-1,0,0,
                            0,0,-1,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            elif allclose_xmin:
                transformed_mesh = mesh.apply_transform(
                        np.array([0,0,-1,0,
                            0,1,0,0,
                            1,0,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            elif allclose_xmax:
                transformed_mesh = mesh.apply_transform(
                        np.array([0,0,1,0,
                            0,1,0,0,
                            -1,0,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
                #  raise NotImplementedError
                pass
            elif allclose_ymin:
                transformed_mesh = mesh.apply_transform(
                        np.array([1,0,0,0,
                            0,0,-1,0,
                            0,1,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            elif allclose_ymax:
                transformed_mesh = mesh.apply_transform(
                        np.array([1,0,0,0,
                            0,0,1,0,
                            0,-1,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            else:
                #  raise NotImplementedError
                pass

            return True, transformed_mesh
        else:
            return False, None

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

        command = "/home/mmf159/Documents/bullet_learning/hello {}".format(" ".join(tmp_objnames))
        self.logger.debug(command)
        command_out = os.system(command)

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







