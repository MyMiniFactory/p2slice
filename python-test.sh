upload_process_dir="upload_process_P2Slice"

if [ ! -d $upload_process_dir ];
then
    echo "${upload_process_dir} doesn't exists, exiting";
    exit 1;
fi

export PYTHONPATH=$PYTHONPATH:${PWD}upload_process_dir

python3 -m unittest test.test_MeshTweaker -v
