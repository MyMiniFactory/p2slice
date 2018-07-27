import os
import numpy as np
import trimesh
import time
from utils import (
    LOG_FORMAT,
    ProcessBase,
    trimesh_load_clean
)

class ProcessWellArranged(ProcessBase):
    def __init__(self):
        super().__init__("WellArranged")

    def perform(self, mesh_path):
        super().perform()

        my_mesh = trimesh_load_clean(mesh_path)
        meshes = my_mesh.split(only_watertight=False)
        number_of_meshes = len(meshes)

        if number_of_meshes <= 1:
            return False

        well_arranged, transformed_mesh = self.is_well_arranged(my_mesh, meshes)
        
        if well_arranged:
            self.logger.debug("Is well arranged")
            ### WARNING : WE CHANGE MY_MESH ###
            transformed_mesh.export(mesh_path)
            return True
        
        self.logger.debug("Is not well aranged")
        return False


        
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
