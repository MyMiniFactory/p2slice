# P2Slice

P2Slice prepares a STL file for FDM 3D Printing.

## Usage

IMPORTANT: Assuming the current working path is root of P2Slice project from now on

```
docker run --rm \
    -v $PWD/p2slice_test:/app/p2slice_test \ # share the stls
    p2slice \ # docker image name is p2slice
        --original_mesh_path ./p2slice_test/no-union-test.stl \ # the input stl
        --pythonpath . \ # indicates where is the py files important
        --error_path ./p2slice_test/error.log \ # log file
        --tmp_path ./p2slice_test/tmp \ # all the output files sit in this directory
        --meshlab_command meshlab \ # not used
        --enable_union \ # do Blender Union
        --verbose # spitting bars at ./p2slice_test/error.log
```

## Installing
```
./docker-build.sh
```

## Running on example files
```
./docker-test-run.sh
```

## See logs by
```
tail -f ./p2slice_test/error.log
```

## Explanation

The "main" file is
```
./upload_process_P2Slice/upload_process_P2Slice.py
```

It includes all the preparations we apply to a STL file, listed below with
explanation.

### clean_file (ProcessCleanFile)

Open the ASCII/binary stl and save to binary STL format

STLReader.py is adapted from STLReader.py in CuraEngine

[Example](./p2slice_test/split-test.stl)

### parse_data (ProcessParseData)

Get mesh information from original stls

### split (ProcessSplit)

Split a STL file to several stls by looking at bounding box (axis-align) of the
connected components

[Example](./p2slice_test/split-test.stl)

### tightly_arranged (ProcessTightlyArranged)

Use Bullet Physics Simulation Engine to test
whether a file is tightly_arranged i.e. multiple models are put in a STL file
and arrange very tightly such that the bounding box (axis-align) are intersecting.

[Example](./p2slice_test/tightly-arranged.stl)

### union (ProcessUnion)

Use Blender apply CSG Operation union to a STL to try to merge several shell to
single shell

[Example](./p2slice_test/union-test.stl)

### shell_count (ProcessShellCount)

calculate percentage of shell with largest number of triangle
/ total number of triangle

Used for deciding whether this STL is click-and-print-able in C&P frontend

(P.S. the pronunciation is /ˈklɪk.ænd.prɪnt.ə.bəl/ not /ˈklɪk.ænd.prɪntə.bəl/)

### tweak (ProcessTweak)

"Improved" version of [Tweaker](https://github.com/ChristophSchranz/Tweaker-3)

### find_best_rect (ProcessFindBestRectangle)

Find the best 2D bounding rectangle for the model

Used for bed arrangement in C&P frontend

### parse_data (ProcessParseData)

Get mesh information from P2Slice stl

### generate_support (ProcessGenerateSupport)

Generate suppport for P2Slice stl

### parse_data (ProcessParseData)

Get mesh information from support stl

## An example

If you run the command at Usage, it should output the following files at ```./p2slice_test/tmp```

```process_report.json``` A report to see whether each process is successful

```no-union-test.json no-union-test_metadata.json```
information of original file

```P2Slice_no-union-test_0_split.json P2Slice_no-union-test_0_split_metadata.json```
information of P2Slice file

``` support_P2Slice_no-union-test_0_split.json support_P2Slice_no-union-test_0_split_metadata.json ```
information of support file

```P2Slice_no-union-test_0_split.stl``` The P2Slice STL

```support_P2Slice_no-union-test_0_split.stl``` support STL for P2Slice file

```tmp_P2Slice_no-union-test.stl``` A copy of the original file not useful

