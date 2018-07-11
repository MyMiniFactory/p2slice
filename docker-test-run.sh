docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/no-union-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/union-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/split-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/simplify-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/tightly-arranged.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/no-tightly-arranged.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json
