from abc import ABC, abstractmethod
from typing import Any
from synthesis.synthesizer.tf_to_torch.torch_enumerator import *
from synthesis.synthesizer.synthesizer import *
from autotesting.auto_test_generation.single_api_test_generation import *
from mapping.representations import *
from synthesis.search_structure import *
from commons.test_utils import execute_api_call, extract_api_arguments, eval_forward_pass_torch
from commons.library_api import load_apis


class SpecInference:

    def __init__(self,  source_library: str, n_tests: int = 100, n_trials: int = 100):
        self.apis = load_apis(source_library)
        self.n_tests = n_tests
        self.np_arrays = []

        for i in range(n_trials):
            dim_x = np.random.randint(1, 50)
            dim_y = 51 - dim_x
            dim_z = np.random.randint(1, 5)
            self.np_arrays.append(np.random.rand(1, dim_x, dim_y, dim_z))

    def infer_spec(self, api_full_name) -> Optional[List[Constraint]]:
        api = next(filter(lambda x: x.id == api_full_name, self.apis))
        tests = self.create_testbed(api)
        return None

    def create_testbed(self, api: LibraryAPI):
        var_pool = VarPool.get_preset_pool()
        tree = OneToOne(api, var_pool)
        test_pool = []
        while tree.has_next() and len(test_pool) < self.n_tests:
            fn_call = tree.next()
            test_case = self.create_test(api, fn_call)
            test_pool += test_case

        return test_pool

    def create_test(self, api: LibraryAPI, fn_call: str) -> List[TestCase]:
        src_api_args = extract_api_arguments(api, fn_call)
        success, layer = execute_api_call(api, src_api_args)
        if success:
            for inpt in self.np_arrays:
                success, output = eval_forward_pass_torch(layer, inpt)
                if success:
                    return [TestCase({0: inpt}, output, api)]
        return []


if __name__ == '__main__':

    inf = SpecInference('torch')
    inf.infer_spec('torch.nn.Conv2d')