import os
import torch
import torch.nn as nn
import unittest
import math
import importlib
import glob
import shutil
try:
    from gradescope_utils.autograder_utils.decorators import weight
    from gradescope_utils.autograder_utils.files import check_submitted_files
except:
    # Decorator which does nothing
    def weight(n):
        return lambda func: lambda *args, **kwargs: func(*args, **kwargs)

def delayed_import(module_name, global_var_name):
    # Use importlib.import_module to handle both top-level modules and submodules
    globals()[global_var_name] = importlib.import_module(module_name)
    
def delayed_import_function_or_class(module_name, object_name, global_var_name):
    # Import the module
    module = importlib.import_module(module_name)    
    # Get the specific function or class from the module
    globals()[global_var_name] = getattr(module, object_name)

def delayed_imports():
    # This has the effect of importing the modules as if we had called:
    # from lora import LoRALinear, mark_only_lora_as_trainable
    delayed_import('lora', 'lora')
    delayed_import_function_or_class('lora', 'LoRALinear', 'LoRALinear')
    delayed_import_function_or_class('lora', 'mark_only_lora_as_trainable', 'mark_only_lora_as_trainable')

def init_output(in_features=10, out_features=5, lora_rank=0, lora_alpha=0.0, lora_dropout=0.0):
    lora_layer = LoRALinear(in_features=in_features, out_features=out_features, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    properties = [
        lora_layer.in_features,
        lora_layer.out_features,
        lora_layer.bias is not None,
        lora_layer.has_weights_merged is False,
        lora_layer.is_lora(),
        hasattr(lora_layer, 'lora_A'),
        hasattr(lora_layer, 'lora_B'),
        hasattr(lora_layer, 'lora_scaling'),
        hasattr(lora_layer, 'lora_dropout')
    ]
    if lora_rank > 0:
        # Extract dropout probability regardless of the type
        if isinstance(lora_layer.lora_dropout, nn.Dropout):
            lora_dropout_p = lora_layer.lora_dropout.p
        else:
            lora_dropout_p = lora_layer.lora_dropout
        new_properties = [
            lora_layer.lora_A.shape,
            lora_layer.lora_B.shape,
            lora_dropout_p,
            lora_layer.lora_scaling,
            lora_layer.lora_A.requires_grad,
            lora_layer.lora_B.requires_grad
        ]
        properties += new_properties
        properties += [lora_layer.lora_A.data, lora_layer.lora_B.data]
    return properties

def reset_params_output(in_features=10, out_features=5, lora_rank=0, lora_alpha=0.0, lora_dropout=0.0):
    lora_layer = LoRALinear(in_features=in_features, out_features=out_features, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    lora_layer.reset_parameters()
    return lora_layer.lora_A.data, lora_layer.lora_B.data

def mark_lora_trainable_output(in_features=10, out_features=5, bias=True, lora_rank=0, lora_alpha=0.0, lora_dropout=0.0):
    lora_layer = LoRALinear(in_features=in_features, out_features=out_features, bias=bias, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    mark_only_lora_as_trainable(lora_layer)
    return { name : param.requires_grad for name, param in lora_layer.named_parameters() }

def forward_output(in_features=10, out_features=5, has_weight_merged=True, lora_rank=0, lora_alpha=0, lora_dropout=0, forward_input=None, lora_A=None, lora_B=None):
    lora_layer = LoRALinear(in_features=in_features, out_features=out_features, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=0)
    lora_layer.has_weight_merged = has_weight_merged
    if lora_rank > 0:
        lora_layer.lora_A = nn.parameter.Parameter(lora_A)
        lora_layer.lora_B = nn.parameter.Parameter(lora_B)
    return lora_layer.forward(forward_input)

def train_output(in_features=10, out_features=5, has_weight_merged=True, lora_rank=0, lora_alpha=0, lora_dropout=0, lora_A=None, lora_B=None):
    lora_layer = LoRALinear(in_features=in_features, out_features=out_features, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    lora_layer.has_weight_merged = has_weight_merged
    if lora_rank > 0:
        lora_layer.lora_A = nn.parameter.Parameter(lora_A)
        lora_layer.lora_B = nn.parameter.Parameter(lora_B)
    lora_layer.train()
    return lora_layer.weight.data, lora_layer.has_weight_merged

def eval_output(in_features=10, out_features=5, has_weight_merged=True, lora_rank=0, lora_alpha=0, lora_dropout=0, lora_A=None, lora_B=None):
    lora_layer = LoRALinear(in_features=in_features, out_features=out_features, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    lora_layer.has_weight_merged = has_weight_merged
    if lora_rank > 0:
        lora_layer.lora_A = nn.parameter.Parameter(lora_A)
        lora_layer.lora_B = nn.parameter.Parameter(lora_B)
    lora_layer.eval()
    return lora_layer.weight.data, lora_layer.has_weight_merged

testcases_dict = torch.load("data.pt", weights_only=True)

class TestLora(unittest.TestCase):
    
    @weight(1)
    def test_01_submitted_files(self):
        """[T01] Check submitted files"""
        if os.path.exists('/autograder/submission'):
            # We are running on Gradescope
            print('Submitted files: ', end='')
            print([x.replace('/autograder/submission/', '') for x in
                glob.glob('/autograder/submission/**/*', recursive=True)])
            required_files = ['lora.py', 'model.py', 'dataloader.py', 'train.py', 'generate.py']
            missing_files = check_submitted_files(required_files)
            assert len(missing_files) == 0, f"Missing files: {missing_files}"
            for file in required_files:
                shutil.copy(f'/autograder/submission/{file}', f'./{file}')
        delayed_imports()

    @weight(1)
    def test_02_reset_parameters(self): 
        global testcases_dict
        torch.manual_seed(2024)
        for testcase in testcases_dict["reset_params"]:
            args = testcase['args']
            expected_lora_A = testcase['expected_lora_A']
            expected_lora_B = testcase['expected_lora_B']
            lora_A, lora_B = reset_params_output(*args)
            torch.testing.assert_close(expected_lora_A, lora_A, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(expected_lora_B, lora_B, atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_03_init(self): #TODO: See if you can isolate reset_parameters
        global testcases_dict
        torch.manual_seed(2024)
        for testcase in testcases_dict["init"]:
            args = testcase['args']
            expected_output = testcase['expected_output']
            output = init_output(*args)
            assert len(expected_output)==len(output) and all([expected==out for expected, out in zip(expected_output[:-2], output[:-2])])
            torch.testing.assert_close(expected_output[-1], output[-1], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(expected_output[-2], output[-2], atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_04_mark_only_lora_as_trainable(self):
        global testcases_dict
        lora_params = ['lora_A', 'lora_B']
        for testcase in testcases_dict["lora_trainable"]:
            args = testcase['args']
            expected_output = testcase['expected_output']
            output = mark_lora_trainable_output(*args)
            assert all([param in output and output[param]==expected_output[param] for param in lora_params])
            #Student can initialize other parameters
            assert all([output[param]!=expected_output['lora_A'] for param in output if param not in lora_params])

    @weight(1)
    def test_05_forward(self):
        global testcases_dict
        for i, testcase in enumerate(testcases_dict["forward"]):
            torch.manual_seed(2024)
            args = testcase['args']
            forward_input = testcase['input']
            lora_A_input = testcase['lora_A']
            lora_B_input = testcase['lora_B']
            expected_output = testcase['expected_output']
            output = forward_output(*args, forward_input=forward_input, lora_A=lora_A_input, lora_B=lora_B_input)
            torch.testing.assert_close(expected_output, output, atol=1e-4, rtol=1e-4, msg="Incorrect")


    @weight(1)
    def test_06_train(self):
        global testcases_dict
        for testcase in testcases_dict["train"]:
            torch.manual_seed(2024)
            args = testcase['args']
            expected_output = testcase['expected_output']
            expected_state = testcase['expected_state']
            lora_A_input = testcase['lora_A']
            lora_B_input = testcase['lora_B']
            output, state = train_output(*args, lora_A=lora_A_input, lora_B=lora_B_input)
            torch.testing.assert_close(expected_output, output, atol=1e-4, rtol=1e-4)
            
            assert state == expected_state
    
    @weight(1)
    def test_07_eval(self):
        global testcases_dict
        for testcase in testcases_dict["eval"]:
            torch.manual_seed(2024)
            args = testcase['args']
            expected_output = testcase['expected_output']
            expected_state = testcase['expected_state']
            lora_A_input = testcase['lora_A']
            lora_B_input = testcase['lora_B']
            output, state = eval_output(*args, lora_A=lora_A_input, lora_B=lora_B_input)
            torch.testing.assert_close(expected_output, output, atol=1e-4, rtol=1e-4)
            assert state == expected_state

if __name__ == "__main__":
    unittest.main()
