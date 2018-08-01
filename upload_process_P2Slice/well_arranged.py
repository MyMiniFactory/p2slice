import os
import numpy as np
import trimesh
import time
import copy
from collections import Counter
from utils import (
    LOG_FORMAT,
    ProcessBase,
    trimesh_load_clean,
    append_data_to_json
)

class ProcessWellArranged(ProcessBase):
    def __init__(self):
        super().__init__("WellArranged")

    def perform(self, mesh_path, P2Slice_json):
        super().perform()

        def calculateArea(m, index):
            [i1, i2 , i3] = m.faces[index]
            p1, p2, p3 = m.vertices[i1], m.vertices[i2], m.vertices[i3]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            return np.linalg.norm(n)/2
            

        my_mesh = trimesh_load_clean(mesh_path)
        meshes = my_mesh.split(only_watertight=False)
        number_of_meshes = len(meshes)

        if number_of_meshes <= 1:
            return False

        orient = Counter()
        align = my_mesh.face_normals
        for index in range(len(align)):       # Cumulate areavectors
            orient[tuple(align[index])] += calculateArea(my_mesh, index)
        orientations = orient.most_common(100)
        
        for m in meshes:
            for ori in orientations:
                ori = np.array(ori[0])
                print(ori)
                ori = ori / np.linalg.norm(ori)
                well_arranged, transformed_mesh = self.is_well_arranged(my_mesh, meshes, ori)
                if well_arranged:
                    self.logger.debug("Is well arranged")
                    transformed_mesh.export(mesh_path)
                    data = {'printInPlace': True}
                    append_data_to_json(data, P2Slice_json)
                    return True
        
        self.logger.debug("Is not well aranged")
        data = {'printInPlace': False}
        append_data_to_json(data, P2Slice_json)
        return False

    @staticmethod
    def applyTransform(mesh, n):
        #n is the orientation vector
        z =  np.array([0,0,1])
        if np.array_equal(z, n):
            A = np.array([1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1]).reshape(4, 4)
        elif np.array_equal(-z, n):
            A = np.array([1,0,0,0,
                0,-1,0,0,
                0,0,-1,0,
                0,0,0,1]).reshape(4, 4)
        else:
            m = np.cross(n, z)
            m = m / np.linalg.norm(m)
            l = np.cross(m, n)
            l = l / np.linalg.norm(l)
            l3 = np.append(n,[0])
            l2 = np.append(m,[0])
            l1 = np.append(l,[0])
            l4 = np.array([0,0,0,1])
            A = np.transpose(np.array([l1, l2, l3, l4]).reshape(4, 4))
            A = np.linalg.inv(A)
        transformed_mesh = copy.deepcopy(mesh).apply_transform(A)
        return transformed_mesh



    def is_well_arranged(self, mesh, list_of_meshes, ori):

        # number based on 12 cm machine where 0.01*12 = 0.01 which is like one layer height
        # assuming your scale the object for small printer (like STARTT)
        ACCEPTED_BASE_PERCENTAGE = 0.01

        meshes = []
        for m in list_of_meshes:
            meshes.append(ProcessWellArranged.applyTransform(m, ori))


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

            transformed_mesh = ProcessWellArranged.applyTransform(mesh, ori)
            if allclose_zmin:
                transformed_mesh = transformed_mesh
            elif allclose_zmax:
                transformed_mesh.apply_transform(
                        np.array([1,0,0,0,
                            0,-1,0,0,
                            0,0,-1,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            elif allclose_xmin:
                transformed_mesh.apply_transform(
                        np.array([0,0,-1,0,
                            0,1,0,0,
                            1,0,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            elif allclose_xmax:
                transformed_mesh.apply_transform(
                        np.array([0,0,1,0,
                            0,1,0,0,
                            -1,0,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
                #  raise NotImplementedError
                pass
            elif allclose_ymin:
                transformed_mesh.apply_transform(
                        np.array([1,0,0,0,
                            0,0,-1,0,
                            0,1,0,0,
                            0,0,0,1]).reshape(4, 4)
                        )
            elif allclose_ymax:
                transformed_mesh.apply_transform(
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
