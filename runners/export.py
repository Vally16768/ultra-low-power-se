inference:
  stream_io:
    state_in:  ["s1_in","s2_in"]
    state_out: ["s1_out","s2_out"]
export:
  onnx:
    dynamic_axes:
      "s1_in":  {0: "B"}
      "s1_out": {0: "B"}
      "s2_in":  {0: "B"}
      "s2_out": {0: "B"}
    inputs:
      - name: "noisy"
      - name: "s1_in"
      - name: "s2_in"
    outputs:
      - name: "enhanced"
      - name: "s1_out"
      - name: "s2_out"
