import onnx
from onnx import helper, TensorProto
import numpy as np

# Create a tiny test model with some weights (like an LLM would have)
# This simulates the structure without the size

# Input: [batch, seq, hidden]
X = helper.make_tensor_value_info('input', TensorProto.FLOAT16, [1, 4, 256])

# Weight tensors (simulating model weights)
W1_data = np.random.randn(256, 256).astype(np.float16)
W1 = helper.make_tensor('weight1', TensorProto.FLOAT16, [256, 256], W1_data.tobytes(), raw=True)

W2_data = np.random.randn(256, 256).astype(np.float16)
W2 = helper.make_tensor('weight2', TensorProto.FLOAT16, [256, 256], W2_data.tobytes(), raw=True)

# Output
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT16, [1, 4, 256])

# Nodes: input -> matmul(W1) -> relu -> matmul(W2) -> output
matmul1 = helper.make_node('MatMul', ['input', 'weight1'], ['hidden1'])
relu = helper.make_node('Relu', ['hidden1'], ['hidden2'])
matmul2 = helper.make_node('MatMul', ['hidden2', 'weight2'], ['output'])

# Graph
graph = helper.make_graph(
    [matmul1, relu, matmul2],
    'test_model',
    [X],
    [Y],
    [W1, W2]  # Initializers (weights)
)

# Model
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
model.ir_version = 8

onnx.save(model, 'test_small_fp16.onnx')
print("Created test_small_fp16.onnx with embedded weights")
print(f"Model has {len(graph.initializer)} initializers (weights)")
