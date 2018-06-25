import unittest
from upload_process_P2Slice import MeshTweaker
from utils import ProcessTweak
from stl import mesh
import numpy as np

class TestTweaker(unittest.TestCase):

    @staticmethod
    def tweakMatrixFromPath(path):
        numpy_stl_mesh = mesh.Mesh.from_file(path)
        my_mesh_for_tweaker = ProcessTweak.preparemeshtotweaker(numpy_stl_mesh)
        result = MeshTweaker.Tweak(my_mesh_for_tweaker)
        return result.Matrix

    def test_Tweak0(self):
        mesh_path = "./p2slice_test/no-union-test.stl";
        self.assertTrue(
            np.allclose(self.tweakMatrixFromPath(mesh_path), np.identity(3))
        )

    def test_Tweak1(self):
        mesh_path = "./p2slice_test/simplify-test.stl"
        self.assertTrue(
            np.allclose(self.tweakMatrixFromPath(mesh_path), np.identity(3))
        )

    def test_Tweak2(self):
        mesh_path = "./p2slice_test/TT_bowl.stl"
        self.assertTrue(
            np.allclose(
                self.tweakMatrixFromPath(mesh_path),
                [
                    [1,  0, 0],
                    [0,  0, 1],
                    [0, -1, 0]
                ]
            )
        )
