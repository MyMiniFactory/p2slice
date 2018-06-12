docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/union-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/split-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union

docker run -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/simplify-test.stl --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union
