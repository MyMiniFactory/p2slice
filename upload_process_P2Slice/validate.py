""" this file takes the process report and make sure everything works """
import json
import os

def all_success(d):
    return all([v['success'] for v in d.values()])

def exists_not_empty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def validate(json_path):
    with open(json_path) as f:
        report = json.load(f)

    # Test all success is true
    assert all_success(report)

    # TODO: the following code runs only if return paths is empty
    # all paths exists and not empty
    for process in report.values():
        if process["data"]:
            for k, v in process["data"].items():
                if k == "paths":
                    for path in v:
                        assert exists_not_empty(path), "{} doesn't exists".format(path)




    # make sure last one is "full P2Slice Process"
    assert "full P2Slice Process" in [
        i["name"] for i in report.values()
    ]
    #  print(report)

    print("good")


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        raise ValueError( "path to process report should be first argument")
    else:
        json_path = sys.argv[1]
        validate(json_path)
