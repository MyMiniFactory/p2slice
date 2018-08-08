filesToTest="
    no-union-test.stl
    union-test.stl
    split-test.stl
    tightly-arranged.stl
    tightly-arranged-1.stl
    no-tightly-arranged-1.stl
    simplify-test.stl
"

for file in $filesToTest; do
    echo Running on "$file" results
    docker run --rm -v $PWD/p2slice_test:/app/p2slice_test -it p2slice --original_mesh_path ./p2slice_test/$file --pythonpath . --error_path ./p2slice_test/error.log --tmp_path ./p2slice_test/tmp --meshlab_command meshlab --enable_union --verbose >/dev/null 2>&1
    python3 ./upload_process_P2Slice/validate.py p2slice_test/tmp/process_report.json
done
