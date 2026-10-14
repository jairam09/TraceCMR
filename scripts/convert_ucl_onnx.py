"""Convert UCL's published mouse cardiac model to ONNX, so TraceCMR can run it without TensorFlow, and check
that the ONNX model gives the same answers.

    python scripts/convert_ucl_onnx.py UCL_REPO_DIR OUT_DIR [Mouse1.h5 Mouse40.h5 ...]

UCL_REPO_DIR holds the authors' unet3plus.py, LICENSE and ModelWeights/SEG459.h5 (github.com/mrphys/
Open-Source_Pre-Clinical_Segmentation, GPL-3.0). OUT_DIR gets:

    ucl_seg459.onnx      the network's main output only (the deep-supervision heads are training aids)
    ucl_model.json       file name, size, SHA-256, where it came from, the tool versions, and the check results
    UCL-LICENSE.txt      the authors' licence (GPL-3.0), which the converted model keeps
    NOTICE.md            what this file is and how it was made

The check runs both models on every slice of the given UCL mice (ED and ES frames, the same preprocessing as
the app) and fails if any probability differs by more than 1e-3 or any pixel gets a different label.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

WEIGHTS_SHA = "3aac032c3725e843736eb01595131bddd64d34bf12d5a02dba16b177048e0d92"
ONNX_NAME = "ucl_seg459.onnx"


def build_keras(repo: Path):
    import tensorflow as tf

    sys.path.insert(0, str(repo))
    from unet3plus import unet3plus

    inp = tf.keras.Input(shape=[256, 256, 1], name="image")
    net = unet3plus(inp, rank=2, add_dropout=1, dropout_rate=0.3, filters=[32, 64, 128, 256, 512], kernel_size=3,
                    encoder_block_depth=2, decoder_block_depth=1, out_channels=3, out_activation="softmax",
                    deep_supervision=True, CGM=False)
    model = tf.keras.Model(inputs=inp, outputs=net.outputs())
    model.load_weights(str(repo / "ModelWeights" / "SEG459.h5"))
    return tf.keras.Model(inputs=model.inputs, outputs=model.outputs[0])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    mice = [Path(p) for p in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    w = repo / "ModelWeights" / "SEG459.h5"
    if sha256(w) != WEIGHTS_SHA:
        print(f"{w} is not the authors' published weights (SHA-256 mismatch)", file=sys.stderr)
        return 1
    import numpy as np
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    keras_model = build_keras(repo)
    spec = (tf.TensorSpec((None, 256, 256, 1), tf.float32, name="image"),)
    onnx_path = out / ONNX_NAME
    tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=17, output_path=str(onnx_path))
    onnx.checker.check_model(str(onnx_path))

    # same answers? (the app's preprocessing: zoom to 0.1 mm, centre 256 x 256, min-max per slice)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tracecmr import autocontour as ac
    from tracecmr.app.session import Session

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    worst, disagree, n_pix = 0.0, 0, 0
    checked = []
    for m in mice:
        s = Session.from_ucl(m)
        for t in (s.ed_frame, s.es_frame):
            pad = ac._fit(s.image[..., t], 256)
            x = np.stack([ac._minmax(z) for z in np.moveaxis(pad.arr, 2, 0)])[..., None].astype(np.float32)
            p_tf = np.asarray(keras_model.predict(x, verbose=0))
            p_on = sess.run(None, {name: x})[0]
            worst = max(worst, float(np.abs(p_tf - p_on).max()))
            disagree += int((p_tf.argmax(-1) != p_on.argmax(-1)).sum())
            n_pix += p_tf.shape[0] * p_tf.shape[1] * p_tf.shape[2]
        checked.append(m.stem)
        print(f"checked {m.stem}: worst |difference| so far {worst:.2e}, label differences {disagree}", flush=True)
    ok = worst <= 1e-3 and disagree == 0
    info = {
        "file": ONNX_NAME, "size": onnx_path.stat().st_size, "sha256": sha256(onnx_path),
        "source": "github.com/mrphys/Open-Source_Pre-Clinical_Segmentation ModelWeights/SEG459.h5 (Shah et al., "
                  "J Cardiovasc Magn Reson 2026)",
        "source_sha256": WEIGHTS_SHA, "license": "GPL-3.0 (the authors' licence, UCL-LICENSE.txt)",
        "input": "float32 (n, 256, 256, 1): short-axis slices at 0.1 mm, min-max scaled per slice",
        "output": "float32 (n, 256, 256, 3): softmax probabilities (background, myocardium, blood pool)",
        "tools": {"tensorflow": tf.__version__, "tf2onnx": tf2onnx.__version__, "onnx": onnx.__version__,
                  "onnxruntime": ort.__version__, "opset": 17},
        "check": {"mice": checked, "max_abs_probability_difference": worst, "pixels": n_pix,
                  "label_differences": disagree, "passed": ok},
    }
    (out / "ucl_model.json").write_text(json.dumps(info, indent=2) + "\n")
    shutil.copyfile(repo / "LICENSE", out / "UCL-LICENSE.txt")
    (out / "NOTICE.md").write_text(
        "# UCL mouse cardiac segmentation model (ONNX)\n\n"
        "This is the trained network published by Shah, Stuckey et al. (J Cardiovasc Magn Reson 2026) at "
        "https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation (ModelWeights/SEG459.h5), converted "
        "unchanged to ONNX by TraceCMR's scripts/convert_ucl_onnx.py so that it runs without TensorFlow. Only the "
        "network's main output is kept. The weights are the authors' work and remain under their licence, "
        "GPL-3.0 (UCL-LICENSE.txt). The source of this file is the authors' repository above together with the "
        "conversion script. TraceCMR itself is MIT-licensed and does not include the authors' code.\n")
    print(json.dumps(info, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
