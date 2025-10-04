# deploy/run_onnx.py
import onnxruntime as ort

def make_session(path: str, intra_threads=1, inter_threads=1, graph_optim="all"):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    so.graph_optimization_level = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ext":      ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[graph_optim]
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
