import sys, argparse
import os
import logging

import numpy as np
from stl import mesh
import json
import time

from utils import print_message_time, LOG_FORMAT, empty_file, append_data_to_json, basename_no_extension
from utils import ProcessParseData, ProcessShellCount, ProcessFindBestRectangle, ProcessGenerateSupport
from utils import ProcessTweak, ProcessSimplify, ProcessCleanFile, ProcessReturnPaths, ProcessReturnOriginalPaths
from utils import ProcessBase
from split import ProcessSplit
from union import ProcessUnion
from tightly_arranged import ProcessTightlyArranged
from well_arranged import ProcessWellArranged

logging.getLogger("trimesh").setLevel(logging.ERROR)

class ProcessP2Slice(ProcessBase):
    def __init__(self):
        super().__init__("full P2Slice Process")

    def perform(self, *args):
        super().perform()
        self.upload_process(*args)

    @staticmethod
    def upload_process(original_mesh_path, result_mesh_path, tmp_path, mesh_name, P2Slice_path,
            _json, metadata_json,
            process_report_file_path, enable_union, meshlab_command):
        """
        :param my_mesh: stl mesh.
               original_mesh: numpy-stl mesh:
               result_mesh_path: string: path to result P2Slice mesh file
               mesh_name: string: fullname of the mesh file
               P2Slice_path: string: list of z-minimum for these faces
               _json: path to original mesh json include data for original mesh
               metadata_json: path to original mesh metadata json
               P2Slice_json: path to P2Slice json
               P2Slice_metadata_json: path to P2Slice Metadata json

        :return None
        """
        #append_data_to_json({"UEUEUEUEUEUE": splited_parts_counter}, metadata_json_path)

        def processSingleMesh(
                in_mesh_path, result_mesh_path, mesh_name, verbose, P2Slice_path, tmp_path, # simplify
                support_free, P2Slice_json, P2Slice_metadata_json, # tweak
                result_support_mesh_path, # generate support
                support_json_path, support_metadata_json, # parse data for support
                meshlab_command
            ):

            # TODO: DRY repated mesh_name
            union(result_mesh_path, P2Slice_metadata_json)
            shell_count(result_mesh_path, P2Slice_json, P2Slice_metadata_json)
            simplify(result_mesh_path, mesh_name, verbose, P2Slice_path, tmp_path, meshlab_command)
            is_well_arranged = well_arranged(result_mesh_path)

            if not is_well_arranged:
                tweak(result_mesh_path, support_free, P2Slice_json, P2Slice_metadata_json)

            find_best_rect(result_mesh_path, P2Slice_metadata_json)
            parse_data(result_mesh_path, P2Slice_json, P2Slice_metadata_json)
            generate_support(result_mesh_path, result_support_mesh_path)
            parse_data(result_support_mesh_path, support_json_path, support_metadata_json)


        clean_file(original_mesh_path, result_mesh_path)

        parse_data(result_mesh_path, _json, metadata_json)

        export_meshes_pathes = split(
            result_mesh_path, tmp_path, mesh_name, metadata_json, enable_union)

        splited_parts_counter = 0
        for export_mesh_path in export_meshes_pathes:

            for ta_export_mesh_path in tightly_arranged(export_mesh_path, tmp_path, P2Slice_path):
                splited_parts_counter += 1

                [ result_mesh_path,
                result_support_mesh_path,
                P2Slice_json_path,
                P2Slice_metadata_json_path,
                support_json_path,
                support_metadata_json_path ] = return_paths(tmp_path, ta_export_mesh_path)

                processSingleMesh(
                    ta_export_mesh_path, result_mesh_path, mesh_name, verbose, P2Slice_path, tmp_path,
                    support_free, P2Slice_json_path, P2Slice_metadata_json_path, # tweak
                    result_support_mesh_path, # generate support
                    support_json_path, support_metadata_json_path, # parse data for support
                    meshlab_command
                )

        append_data_to_json({"number_of_splited_parts": splited_parts_counter}, metadata_json_path)

parse_data = ProcessParseData().run
shell_count = ProcessShellCount().run
simplify = ProcessSimplify().run
find_best_rect = ProcessFindBestRectangle().run
generate_support = ProcessGenerateSupport().run
tweak = ProcessTweak().run
clean_file = ProcessCleanFile().run
split = ProcessSplit().run
union = ProcessUnion().run
return_paths = ProcessReturnPaths().run
return_original_paths = ProcessReturnOriginalPaths().run
upload_process = ProcessP2Slice().run
tightly_arranged = ProcessTightlyArranged().run
well_arranged = ProcessWellArranged().run

if __name__ == "__main__":
    import slicing_config
    from utils import ProcessBase
    """Get the command line args, ask for the execution of the process"""
    parser = argparse.ArgumentParser(description=
                                     "Process to apply on the uploaded file")
    parser.add_argument('--original_mesh_path', action="store", help="select id server of file", required=True)
    parser.add_argument('-v', '--verbose', action="store_true",dest="verbose", help="increase output verbosity", default=False)
    parser.add_argument('--pythonpath', action="store", dest="python_path", help="path to python script", required=True)
    parser.add_argument('--support_free', action="store_true", help="disable tweaker if the file is support free", default=False)
    parser.add_argument('--error_path', action="store",  help="path to P2Slice.log for errors", required=True)
    parser.add_argument('--tmp_path', action="store", help="path to create all files", required=True)
    parser.add_argument('--enable_union', action="store_true", help="do slow union", default=False) # not used
    parser.add_argument('--meshlab_command', action="store", help="full command for calling meshlabserver", default=False, required=True)
    args = parser.parse_args()


    if args.python_path :
        P2Slice_path = args.python_path

    # TODO: use support_free
    verbose = args.verbose
    if verbose:
        ProcessBase.log_level = logging.DEBUG
    else:
        ProcessBase.log_level = logging.WARNING

    ProcessBase.error_log_path = args.error_path

    support_free = args.support_free

    tmp_path = args.tmp_path
    enable_union = args.enable_union
    meshlab_command = args.meshlab_command

    process_report_file_path = os.path.join(tmp_path, "process_report.json")
    empty_file(process_report_file_path)
    ProcessBase.process_report_file_path = process_report_file_path
    
    original_mesh_path = args.original_mesh_path
    mesh_name = basename_no_extension(original_mesh_path)
   
    [ result_mesh_path,
    json_path,
    metadata_json_path,
    ] = return_original_paths(original_mesh_path, tmp_path)

    # preprocess stl file
    upload_process(
        original_mesh_path, result_mesh_path, tmp_path, mesh_name, P2Slice_path,
        json_path, metadata_json_path,
        process_report_file_path,
        enable_union, meshlab_command)
