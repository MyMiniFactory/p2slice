import time
import os
import json
import logging
import random
import scipy
from collections import Counter

import numpy as np
from stl import mesh
import sys, traceback
import trimesh
from rectangle_creation import best_rectangle
import generate_support_material
from stl import mesh
from MeshTweaker import Tweak
from Simplify import simplify
import STLReader

def trimesh_load_clean(mesh_path):
    from trimesh import constants

    m = trimesh.load(
        mesh_path,
        process=True,
        validate=True
    )

    m.remove_degenerate_faces(constants.tol.merge)

    return m


def vertices_split(mesh_path, _type="path"):
    # _type is either "path" or "mesh"
    if _type == "mesh":
        mesh_0 = mesh_path
    else:
        mesh_0 = trimesh_load_clean(mesh_path)

    # split on vertices connectivity
    length = np.max(list(set(mesh_0.faces.flatten())))
    length += 1

    f = mesh_0.faces

    x = f[:,0]
    y = f[:,1]
    z = f[:,2]

    row = np.concatenate((x, x, z))
    col = np.concatenate((y, z, y))
    data = np.ones(len(row))

    connectivity_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(length, length))
    final_connectivity = scipy.sparse.csgraph.connected_components(connectivity_matrix, directed=False)

    group = final_connectivity[1]
    number_of_groups = final_connectivity[0]

    c = Counter(group)

    length_of_group = c.most_common(1)[0][1]

    shells_counter = length_of_group/len(group)

    return shells_counter

# TODO: find a better algo? or return True is actually the best?
def does_accept_part(split_mesh, original_mesh):

    # number_tri_percetage = len(split_mesh.triangles) / len(original_mesh.triangles)
    # number_tri_condition = number_tri_percetage > 0.001

    # area_percentage = split_mesh.area/original_mesh.area
    # area_condition = area_percentage > 0.001

    # self.logger.debug("number_tri_percetage {}".format(number_tri_percetage))
    #  self.logger.debug("area_percentage {}".format(area_percentage))

    # self.logger.debug("number_tri_condition {}".format(number_tri_condition))
    #  self.logger.debug("area_condition {}".format(area_condition))
    #  if number_tri_condition or area_condition:
    # if area_condition:
        # return True
    # else:
        # return False

    return True

def basename_no_extension(path):
    return os.path.splitext(os.path.basename(path))[0]

def print_message_time(message, start_time):
    print('{} took {:.2f} s'.format(message, time.time() - start_time))

def obj_to_stl(obj_path, stl_path, meshlab_command):
    #  command = "{} -i {} -o {} >/dev/null 2>&1".format(
	#  meshlab_command,
        #  obj_path,
        #  stl_path
    #  )
    #  os.system(command)

    m = trimesh.load_mesh(obj_path)
    m.export(stl_path)
    if not os.path.exists(stl_path):
        raise ValueError("meshlabserver error")

def stl_to_obj(stl_path, obj_path, meshlab_command, meshlab_script_path):
    #  os.system("{} -i {} -o {} -s {} >/dev/null 2>&1".format(MESHLABSERVER_COMMAND, stl_path, obj_path, meshlab_script_path))
    #  os.system("{} -i {} -o {} -s {}".format(meshlab_command, stl_path, obj_path, meshlab_script_path))
    m = trimesh.load_mesh(stl_path)
    m.export(obj_path)
    if not os.path.exists(obj_path):
        raise ValueError("meshlab stl to obj creation error")

def save_trimesh_as_obj(mesh, obj_path):
    with open(obj_path, "w") as f:
        f.write('v ' + '\nv '.join(' '.join(str(x) for x in y) for y in mesh.vertices))
        f.write('\nf ' + '\nf '.join(' '.join(str(x) for x in y) for y in mesh.faces + 1))
    if not os.path.exists(obj_path):
        raise ValueError("mesh export to obj error")

LOG_FORMAT = '%(asctime)s %(name)s %(levelname)-7s %(message)s'

def empty_file(filepath):
    with open(filepath, "w+") as json_file:
        json.dump({}, json_file)

def append_data_to_json(data, path_to_json):
    """ the json is a big dictionary,
        we want append data in to this dictionary
    """

    with open(path_to_json, 'r+') as json_file:
        d = json.load(json_file)

    d.update(data)

    with open(path_to_json, 'w+') as json_file:
        json.dump(d, json_file)

module_logger = logging.getLogger("Process")

# TODO: add log decorator
class ProcessBase:

    error_log_path = None
    log_level = None
    process_report_file_path = None

    def __init__(self, name):
        import slicing_config
        self.name = name
        self.data = {}
        self.logger = logging.getLogger("Process "+self.name)

    def update_logger(self):
        logging.basicConfig(format=LOG_FORMAT,
                filename=ProcessBase.error_log_path,
                level=ProcessBase.log_level)


    def perform(self):
        #  self.logger.debug("{} perform".format(self.name))
        pass

    def start(self):
        #  self.logger.debug("{} start".format(self.name))
        pass

    def finish(self):
        # finished can be success or failure
        #  self.logger.debug("{} finish".format(self.name))
        pass

    def error(self):
        #  self.logger.debug("{} error".format(self.name))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        self.logger.error(
            "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        )

        self._process_report(False, self.data)
        sys.exit()

    def success(self):
        #  self.logger.debug("{} success".format(self.name))
        self._process_report(True, self.data)

    def run(self, *args):
        self.update_logger()
        #  self.logger.debug("{} run".format(self.name))
        start_time = time.time()
        self.start()
        try:
            result = self.perform(*args)
        except Exception as inst :
            self.error()
        else:
            self.success()
            #  self.logger.debug("success in {}s".format(time.time() - start_time))

        self.finish()
        return result

    def _process_report(self, success, data):
        message = {"success":success, "name":self.name, "data":data}
        #  self.logger.info(message)
        with open(ProcessBase.process_report_file_path, "r+") as f:
            report = json.load(f)

        next_key = len(report.keys()) + 1
        report[next_key] = message

        with open(ProcessBase.process_report_file_path, 'w+') as json_file:
            json.dump(report, json_file)

    def add_data(self, key, value):
        if key not in self.data:
            self.data[key] = value
        else:
            raise NotImplementedError


class ProcessParseData(ProcessBase):
    def __init__(self):
        super().__init__("Parse Data")

    def perform(self, mesh_path, json_path, metadata_json_path):
        super().perform()
        numpy_stl_mesh = mesh.Mesh.from_file(mesh_path) # minimun rotated bounding box
        self.parsedata(numpy_stl_mesh, json_path, metadata_json_path)

    @staticmethod
    def parsedata(my_mesh, _json, metadata_json) :
        # my_mesh is numpy stl mesh
        # Parse data from stl
        nbr_triangles = len(my_mesh)

        xmin = np.float64(np.amin(my_mesh.x))
        ymin = np.float64(np.amin(my_mesh.y))
        zmin = np.float64(np.amin(my_mesh.z))

        xmax = np.float64(np.amax(my_mesh.x))
        ymax = np.float64(np.amax(my_mesh.y))
        zmax = np.float64(np.amax(my_mesh.z))

        length = 0.0
        surface = 0.0

        # Open mesh with numpy
        mesh = np.array(my_mesh, dtype=np.float64)
        mesh = mesh.reshape(nbr_triangles,3,3)
        # vertices
        v0=mesh[:,0,:]
        v1=mesh[:,1,:]
        v2=mesh[:,2,:]
        length = np.sum( np.sqrt(np.sum((np.subtract(v1,v0))**2,axis=1)) + np.sqrt(np.sum((np.subtract(v2,v0))**2,axis=1)) + np.sqrt(np.sum((np.subtract(v2,v1))**2,axis=1)))
        # compute normals
        normals = np.cross( np.subtract(v1,v0), np.subtract(v2,v0)).reshape(nbr_triangles,3)
        surface = np.sum(np.sqrt(np.sum(normals**2,axis=1)))

        # Calculing average length of edges in the mesh
        average_edge_length = length/(len(my_mesh)*3)
        # Computing average surface of triangles in the mesh
        average_triangle_surface = surface/(len(my_mesh))

        volume, cog, inertia = my_mesh.get_mass_properties()

        metadata = {'average_edge_length': average_edge_length,
                'average_triangle_surface': average_triangle_surface,
                'surface': surface,
                'center_of_gravity': cog.tolist(),
                'inertia_matrix_cog': inertia.tolist()
                }
        append_data_to_json(metadata, metadata_json)

        data = {'bounding_box_x': xmax - xmin,
                'bounding_box_y': ymax - ymin,
                'bounding_box_z': zmax - zmin,
                'triangles_count': nbr_triangles,
                'volume': volume
                }
        append_data_to_json(data, _json)

class ProcessShellCount(ProcessBase):
    def __init__(self):
        super().__init__("Shell Count")

    def perform(self, mesh_path, json, metadata_json):
        super().perform()
        main_mesh_percentage = vertices_split(mesh_path)
        append_data_to_json({'shells_counter': main_mesh_percentage}, json)
        # append_data_to_json({'number_triangles_distribution': number_of_faces}, metadata_json)
        return main_mesh_percentage

class ProcessSimplify(ProcessBase):
    def __init__(self):
        super().__init__("Simplify")

    def perform(self, out_file, mesh_name, verbose, P2Slice_path, tmp_path, meshlab_command):
        super().perform()
        simplified_mesh = simplify(
            out_file, mesh_name, verbose, P2Slice_path, tmp_path, meshlab_command, logging=self.logger)


class ProcessFindBestRectangle(ProcessBase):
    def __init__(self):
        super().__init__("Find Best Rectangle")

    def perform(self, mesh_path, P2Slice_metadata_json):
        super().perform()
        # Compute best bottom bounding box to arrange multipart object on the bed
        rectangle = best_rectangle(mesh_path) # change the mesh
        self.addBestRectangleInJson(rectangle[0],rectangle[1],P2Slice_metadata_json)

    @staticmethod
    def addBestRectangleInJson(theta, dico, P2Slice_metadata_json) :
        data = {
            'best_rectangle': dico["best_rectangle"],
            'best_theta': theta
        }
        append_data_to_json(data, P2Slice_metadata_json)

class ProcessGenerateSupport(ProcessBase):
    def __init__(self):
        super().__init__("Generate Support")

    def perform(self, mesh_path, support_mesh_path):
        super().perform()
        generate_support_material.support_generation_process(mesh_path, support_mesh_path)

class ProcessTweak(ProcessBase):
    def __init__(self):
        super().__init__("Tweak")

    def perform(self, mesh_path, support_free, P2Slice_json, P2Slice_metadata_json):
        super().perform()
        # use tweaker
        # Tweak only if not support_free
        if( not support_free):
            # Change data structure to tweaker
            numpy_stl_mesh = mesh.Mesh.from_file(mesh_path)
            my_mesh_for_tweaker = self.preparemeshtotweaker(numpy_stl_mesh)

            verbose = True
            favside = False
            extended_mode = True

            x = Tweak(my_mesh_for_tweaker, extended_mode, verbose, favside,
                    logging=self.logger)

            self.addMatrixInJson(x, P2Slice_json, P2Slice_metadata_json)
            self.logResult(x)

            # Apply matrix rotation to stl file
            m = np.linalg.inv(x.Matrix)
            transform_matrix = [[m[0][0], m[0][1], m[0][2], 0],
                   [m[1][0], m[1][1], m[1][2], 0],
                   [m[2][0], m[2][1], m[2][2], 0],
                   [0, 0, 0, 1]]
            transform_matrix = np.array(transform_matrix)

            numpy_stl_mesh.transform(transform_matrix)
            numpy_stl_mesh.save(mesh_path)

    @staticmethod
    def preparemeshtotweaker(mesh) :
        # output of stl-mesh is mesh = [triangles] with triangle = [vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
        # input of tweaker is mesh = [triangles] with triangles = [vertices] with vertice = [x, y ,z]
        n = len(mesh)
        points = mesh.points.copy()

        return points.reshape(n*3,3)

    @staticmethod
    def addMatrixInJson(x, P2Slice_json, P2Slice_metadata_json):

        # TODO: if there is only rotation around z axis, it is also not tweaked

        tweaked = 1
        if x.Matrix[0][0] == x.Matrix[1][1] == x.Matrix[2][2] == 1.0 :
            tweaked = 0

        matrix = np.linalg.inv(x.Matrix)
        matrix = matrix.tolist()

        metadata = {
            'rotation_matrix': matrix,
            'alignment': x.Alignment.tolist()
        }
        append_data_to_json(metadata, P2Slice_metadata_json)

        data = {
            'tweaked': tweaked,
            'bottom_area': x.BottomArea,
            'overhang': x.Overhang,
            'tweakermetadata': x.datatweaker,
            'unprintability_rate': x.Unprintability
        }
        append_data_to_json(data, P2Slice_json)

    def logResult(self, x) :
        self.logger.debug("Result-stats:")
        self.logger.debug(" Tweaked Z-axis: \t{}".format((x.Alignment)))
        self.logger.debug(" Axis, angle:   \t{}".format(x.Euler))
        self.logger.debug(" Rotation matrix: {:2f}\t{:2f}\t{:2f} {:2f}\t{:2f}\t{:2f} {:2f}\t{:2f}\t{:2f}".format(x.Matrix[0][0], x.Matrix[0][1], x.Matrix[0][2], x.Matrix[1][0], x.Matrix[1][1], x.Matrix[1][2], x.Matrix[2][0], x.Matrix[2][1], x.Matrix[2][2]))
        self.logger.debug("Unprintability: \t{}".format(x.Unprintability))
        self.logger.debug("Successfully Rotated!")

class ProcessCleanFile(ProcessBase):
    def __init__(self):
        super().__init__("Clean File")

    def perform(self, original_mesh_path, clean_mesh_path):
        super().perform()

        if (os.path.getsize(original_mesh_path) < 100):
            raise Exception('Empty File')
        reader = STLReader.STLReader()
        my_mesh = reader.load_file(original_mesh_path)

        # save the successfull open file
        my_mesh.save(clean_mesh_path)

class ProcessReturnOriginalPaths(ProcessBase):
    def __init__(self):
        super().__init__("Return Original File Paths")

    def perform(self, *args):
        super().perform()
        self.data["paths"] = self.return_paths(*args)
        return self.data["paths"]

    @staticmethod
    def return_paths(original_mesh_path, tmp_path):
        mesh_name = basename_no_extension(original_mesh_path)
        result_mesh_path = os.path.join(tmp_path, 'tmp_P2Slice_{}.stl'.format(mesh_name))

        # json file for inserting data into mysql
        json_path = os.path.join(tmp_path, '{}.json'.format(mesh_name))
        metadata_json_path = os.path.join(tmp_path, '{}_metadata.json'.format(mesh_name))

        empty_file(json_path)
        empty_file(metadata_json_path)

        result = [result_mesh_path, json_path, metadata_json_path]

        return result

class ProcessReturnPaths(ProcessBase):
    def __init__(self):
        super().__init__("Return Paths")

    def perform(self, *args):
        super().perform()
        self.data["paths"] = self.return_paths(*args)
        return self.data["paths"]

    @staticmethod
    def return_paths(tmp_path, export_mesh_path):
        mesh_name = basename_no_extension(export_mesh_path)
        assert("P2Slice" in mesh_name)
        result_mesh_path = os.path.join(tmp_path, '{}.stl'.format(mesh_name))
        result_support_mesh_path = os.path.join(tmp_path, 'support_{}.stl'.format(mesh_name))

        # json file for inserting data into mysql
        P2Slice_json_path = os.path.join(tmp_path, '{}.json'.format(mesh_name))
        P2Slice_metadata_json_path = os.path.join(tmp_path, '{}_metadata.json'.format(mesh_name))

        support_json_path = os.path.join(tmp_path, 'support_{}.json'.format(mesh_name))
        support_metadata_json_path = os.path.join(tmp_path, 'support_{}_metadata.json'.format(mesh_name))

        empty_file(P2Slice_json_path)
        empty_file(P2Slice_metadata_json_path)

        empty_file(support_json_path)
        empty_file(support_metadata_json_path)

        result = [ result_mesh_path,
        result_support_mesh_path,
        P2Slice_json_path,
        P2Slice_metadata_json_path,
        support_json_path,
        support_metadata_json_path ]

        return result

class ProcessTest(ProcessBase):
    def __init__(self):
        super().__init__("Test")

    def perform(self):
        super().perform()
        self.logger.debug("This is a test")

def main():
    from stl import mesh
    my_mesh = mesh.Mesh.from_file("/home/mmf159/Documents/Copies/P2Slice_aries.stl")
    json_path = "/home/mmf159/Documents/test.json"
    json_metadata_path = "/home/mmf159/Documents/test_metadata.json"
    test = ProcessParseData(my_mesh, json_path, json_metadata_path)
    test.perform()

if __name__ == "__main__":
    main()
