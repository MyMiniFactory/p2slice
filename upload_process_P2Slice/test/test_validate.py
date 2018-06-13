import unittest
import tempfile
from validate import *

class TestValidate(unittest.TestCase):

    def test_all_success(self):
        dict0 = {0:{"success":True}, 1:{"success":False}}
        dict1 = {0:{"success":False}, 1:{"success":False}}
        dict2 = {0:{"success":True}, 1:{"success":True}}

        self.assertFalse(all_success(dict0))
        self.assertFalse(all_success(dict1))
        self.assertTrue(all_success(dict2))


    def test_exists_not_empty(self):

        self.assertFalse(exists_not_empty("./not_exists.json"))
        with tempfile.NamedTemporaryFile() as temp:
            self.assertFalse(exists_not_empty(temp.name))

        with tempfile.NamedTemporaryFile() as temp:
            temp.write("tigertigertiger")
            temp.flush()
            self.assertTrue(exists_not_empty(temp.name))


