# Copyright (c) 2015 Ultimaker B.V.
# Copyright (c) 2013 David Braam
# Uranium is released under the terms of the AGPLv3 or higher.

import os
import struct
import numpy

import stl  # numpy-stl lib
import stl.mesh
from stl import mesh

# Increase max count. (10 million should be okay-ish)
stl.stl.MAX_COUNT = 10000000

class MeshReader():
    def __init__(self):
        super().__init__()

    def read(self, file_name):
        raise NotImplementedError("MeshReader plugin was not correctly implemented, no read was specified")


class STLReader(MeshReader):
    def __init__(self):
        super(STLReader, self).__init__()
        self._supported_extensions = [".stl"]

    def load_file(self, file_name):
        try :
            mesh = self._loadWithNumpySTL(file_name)
        except Exception as inst :
            f = open(file_name, "rb")
            mesh = self._loadBinary( f)
            if not mesh:
                f.close()
                f = open(file_name, "rt")
                try:
                    mesh = self._loadAscii( f)
                except UnicodeDecodeError:
                    return None
                f.close()
        return mesh

    def _loadWithNumpySTL(self, file_name):
        loaded_data = stl.mesh.Mesh.from_file(file_name)
        return(loaded_data)


    # Private
    ## Load the STL data from file by consdering the data as ascii.
    # \param f The file handle
    def _loadAscii(self, f):
        num_verts = 0
        for lines in f:
            for line in lines.split("\r"):
                if "vertex" in line:
                    num_verts += 1

        data = numpy.zeros(num_verts, dtype=mesh.Mesh.dtype)
        f.seek(0, os.SEEK_SET)
        vertex = 0
        face = [None, None, None]
        face_count=0
        for lines in f:
            for line in lines.split("\r"):
                if "vertex" in line:
                    face[vertex] = line.split()[1:]
                    vertex += 1
                    if vertex == 3:
                        data['vectors'][face_count] = numpy.array(face)
                        vertex = 0
                        face_count += 1
        return(mesh.Mesh(data, remove_empty_areas=True))


    # Private
    ## Load the STL data from file by consdering the data as Binary.
    # \param f The file handle
    def _loadBinary(self, f):
        f.read(80)  # Skip the header

        num_faces = struct.unpack("<I", f.read(4))[0]
        # On ascii files, the num_faces will be big, due to 4 ascii bytes being seen as an unsigned int.
        if num_faces < 1 or num_faces > 1000000000:
            return None
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(84, os.SEEK_SET)
        if file_size < num_faces * 50 + 84:
            return None

        data = numpy.zeros(num_faces, dtype=mesh.Mesh.dtype)
        for idx in range(0, num_faces):
            data_loaded = struct.unpack(b"<ffffffffffffH", f.read(50))
            data['vectors'][idx] = numpy.array([[data_loaded[3], data_loaded[4], data_loaded[5]],
                                               [data_loaded[6], data_loaded[7], data_loaded[8]],
                                               [data_loaded[9], data_loaded[10], data_loaded[11]]])


        return mesh.Mesh(data, remove_empty_areas=True)
