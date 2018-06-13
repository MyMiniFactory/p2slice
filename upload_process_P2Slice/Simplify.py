import sys, argparse
import os
import logging
import numpy as np
from stl import mesh
import json
import time

# Constants
MIN_FILE_SIZE = 160000.0 # equivalent to 8 Mo

def getargs():
    """
    The possible input arguments
    :return: arguments put in input in the terminal
    """
    parser = argparse.ArgumentParser(description=
                                     "Automatic Simplify\n\n")
    parser.add_argument('-i', action="store", dest="inputfile",
                        help="path to stl file")

    args = parser.parse_args()

    argv = sys.argv[1:]

    return args

def simplify(mesh_path, mesh_name, verbose, simplify_path, TMP_PATH,
        meshlab_command, logging = logging):

    tmp_obj_path = os.path.join(TMP_PATH, mesh_name + "_simplify.obj")

    from utils import stl_to_obj, obj_to_stl

    original_size = os.path.getsize(mesh_path)

    my_mesh = mesh.Mesh.from_file(mesh_path)
    nbr_triangles = len(my_mesh)
    factor = MIN_FILE_SIZE / nbr_triangles
    if factor < 1.0 :

        # Convert stl to obj to use simplify
        stl_to_obj_start_time = time.time()
        try :
            logging.debug('Converting stl to obj ...')

            stl_to_obj_start_time_0 = time.time()
            stl_to_obj(mesh_path, tmp_obj_path, meshlab_command, os.path.join(simplify_path, "remove_duplicated_vertex.mlx"))
        except Exception as inst :
            logging.error(str(inst))
            raise_simplify_error()

        # Simplify
        try :
            simplify_start_time = time.time()
            os.system('{0}/simplify {1} {1} {2} > /dev/null 2>&1'.format(simplify_path, tmp_obj_path, factor))
        except Exception as inst :
            logging.error(str(inst))
            raise_simplify_error()

        # Convert back obj to stl
        try :
            logging.debug('Converting obj to stl ...')
            obj_to_stl_start_time = time.time()
            obj_to_stl(tmp_obj_path, mesh_path, meshlab_command)
            # TODO : avoid to save and open again, find a way to open the mesh for tweaker directly
        except Exception as inst :
            logging.error(str(inst))
            raise_simplify_error()

        new_size = os.path.getsize(mesh_path)

        assert new_size != original_size


        logging.debug('Removing useless files ...')
        if os.path.exists(tmp_obj_path):
            os.remove(tmp_obj_path)
        else:
            raise_simplify_error()

    else :
        logging.debug('{} does not need to be simplified'.format(mesh_name))

def raise_simplify_error():
    raise ValueError("Simplify Error")
