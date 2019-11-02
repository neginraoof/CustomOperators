
# Overview of the Required Steps

  - Implementing the custom operator in C++ and registering it with TorchScript
  - Exporting the custom Operator to ONNX, either:
            - as a combination of existing ONNX ops
            or
            - as a custom ONNX Operator
  - If adding a custom Operator in ONNX, the next step is implementing the Operator in ONNX Runtime

# Implement the Custom Opeator
If you have a custom op that you need to add in PyTorch as a C++ extension, you need to implement the op and build it with ```setuptools```.
Start by implementing the operator in C++. Below we have the example C++ code group norm operator:

```cpp
#include <torch/script.h>

torch::Tensor custom_group_norm(torch::Tensor X, torch::Tensor num_groups, torch::Tensor scale, torch::Tensor bias, torch::Tensor eps) {
  torch::Tensor norm =
      at::group_norm(X, static_cast<int>(num_groups.data<float>()[0]), scale, bias);

  return norm.clone();
}
```
<Headr file details>

Next you need to register this operator with TorchScript compiler using ```torch::RegisterOperator``` function in the same cpp file. The first argument is operator name and namespace separated by ```::```. The next argument is a reference to your function. 

```cpp
static auto registry =
  torch::RegisterOperators("mynamespace::custom_group_norm", &custom_group_norm);
```

Once you have your C++ function, you can build it using ```setuptools.Extension```. Create a ```setup.py script``` in the same directory where you have your C++ code. To have all the required include paths, you can use the ```CppExtension.BuildExtension```. Takes care of the required compiler flags such as required include paths, and flags required during mixed C++/CUDA mixed compilation.

For thix example, we are only providing the forward pass function needed for inferencing. Similarly you can implement the backwards pass if needed.

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_group_norm',
      ext_modules=[cpp_extension.CppExtension('custom_group_norm', ['custom_group_norm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```

<Python binding examples>

Now, running the command ```python setup.py install``` from your source directory, you can to build and install your extension.
The generated shared object should be found under ```build``` directory. You can load it using:
```torch.ops.load_library("<path_to_object_file>)```
And then you can refer to your custom operator using:
```torch.ops.<namespace_name>.<operator_name>```

# Export the Operator to ONNX

You can export you custom operator using existing ONNX ops, or you can create custom ONNX ops to use.
In both cases, you need to add the symbolic method for the exporter, and register your custom symbolic using ```torch.onnx.register_custom_op_symbolic```.
First argument conatins the namespace and operator name, separated by ```::```. You also need to pass a reference to the custom symbolic method, and the opset version to export the model to. You can add your script in a python file under the source directory.
```python
def my_group_norm(g, input, num_groups, scale, bias, eps):
    return g.op("mydomain::mygroupnorm", input, num_groups, scale, bias, epsilon_f=eps)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('mynamespace::custom_group_norm', my_group_norm, 1)
```

In the symbolic method, you need to implement the ONNX subgraph to use for exporting your custom op. If you are using existing ONNX operators (from the default ONNX domain), you don't have to mention the domain name.
In our example, we want to use a custom ONNX op from a custom domain. That's why we need to use a string with the following format:
```"<domain_name>::<onnx_op>"```
Now, in order to be able to use this custom ONNX operator for inferencing, we need to implement this in a backend (?). If you are using existing ONNX ops only, you do not need to go through this last step.


# Implement the Operator in ONNX Runtime #

The last step is to implement this op in ONNX Runtime. We show how to do this using their custom op C API's. [APIs are experimental for now.]
First, you need to create a ```Ort::CustomOpDomain```. This domain should have the same name provided in the symbolic method when exporting the model [with the prefix ```org.pytorch.```. need to fix this.]
```cpp
Ort::CustomOpDomain custom_op_domain("org.pytorch.mydomain");
```
Next, you need to create a custom kernel and  ```ORT::CustomOp``` struct and add it to the custom domain. You can find an example here. 
Once you have the custom kernel and schema, you can add them to the domain using the C API as below:
```cpp
GroupNormCustomOp custom_op;
custom_op_domain.Add(&custom_op);
```

Here you can find our example group norm implementation along with a sample ONNX Runtime unit test to verify the expected output.

================================================================

