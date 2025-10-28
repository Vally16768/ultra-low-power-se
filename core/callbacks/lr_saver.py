# core/callbacks/lr_saver.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras


def _to_float(value: Any) -> float:
    """Best-effort: convert tensors/variables/np scalars/py scalars to float."""
    try:
        # tf.Variable / tf.Tensor path
        if isinstance(value, (tf.Variable, tf.Tensor)):
            return float(tf.keras.backend.get_value(value))
        # numpy scalar
        import numpy as _np  # local import to avoid hard dep if stripped
        if isinstance(value, _np.generic):
            return float(value.item())
        return float(value)
    except Exception:
        # last resort: try K.get_value (covers variables/tensors)
        try:
            return float(tf.keras.backend.get_value(value))
        except Exception:
            # give up, but remain safe
            return float("nan")


def _is_schedule(obj: Any) -> bool:
    """Return True if obj behaves like a Keras learning rate schedule."""
    try:
        from tensorflow.keras.optimizers.schedules import LearningRateSchedule
        if isinstance(obj, LearningRateSchedule):
            return True
    except Exception:
        pass
    return callable(obj)


class LRSaver(keras.callbacks.Callback):
    """
    Log the *current* learning rate(s) into Keras `logs`, so they end up in
    `history.history`. Works with:
      - Constant float LR (optimizer.learning_rate or optimizer.lr)
      - tf.Variable / tf.Tensor LR
      - LearningRateSchedule or any callable LR (expects `step` argument)
      - Multiple optimizers (logs one key per optimizer)
    Set `on="batch"` if you want per-batch LR logging (e.g., warmup schedules).

    Parameters
    ----------
    log_name : str
        Base key under which LR is logged. For multiple optimizers, the keys are
        `<log_name>/<optimizer_name>`.
    on : {"epoch", "batch"}
        Whether to log at the end of each epoch or each batch.
    dtype_for_decayed_lr : tf.dtypes.DType
        Dtype to request when using legacy private API `_decayed_lr` as a fallback.
    """

    def __init__(
        self,
        log_name: str = "learning_rate",
        on: str = "epoch",
        dtype_for_decayed_lr: tf.dtypes.DType = tf.float32,
    ) -> None:
        super().__init__()
        if on not in {"epoch", "batch"}:
            raise ValueError("`on` must be 'epoch' or 'batch'")
        self.log_name = str(log_name)
        self.on = on
        self._dtype_for_decayed_lr = dtype_for_decayed_lr
        self._optimizers: List[Tuple[str, keras.optimizers.Optimizer]] = []

    # --- Keras hooks ---------------------------------------------------------

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # Resolve optimizer(s): Keras usually stores a single optimizer as `self.model.optimizer`,
        # but in some advanced setups it can be a list/dict. We normalize to a list of (name, opt).
        self._optimizers = self._collect_optimizers()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.on != "epoch":
            return
        self._write_lrs_into_logs(logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.on != "batch":
            return
        self._write_lrs_into_logs(logs)

    # --- internals -----------------------------------------------------------

    def _collect_optimizers(self) -> List[Tuple[str, keras.optimizers.Optimizer]]:
        """Return list of (name, optimizer)."""
        opts: List[Tuple[str, keras.optimizers.Optimizer]] = []
        opt_obj = getattr(self.model, "optimizer", None)

        def _name_for(o: Any, idx: int) -> str:
            # Prefer the optimizer's internal name if available
            n = getattr(o, "_name", None) or getattr(o, "name", None)
            if isinstance(n, (str,)):
                return n
            # fallback: class name with index
            return f"{o.__class__.__name__}_{idx}"

        if opt_obj is None:
            return opts

        if isinstance(opt_obj, Mapping):
            for k, v in opt_obj.items():
                if isinstance(v, keras.optimizers.Optimizer):
                    # use the mapping key as name if it is string-like
                    name = str(k)
                    opts.append((name, v))
        elif isinstance(opt_obj, Iterable) and not isinstance(opt_obj, (str, bytes)):
            for i, v in enumerate(opt_obj):
                if isinstance(v, keras.optimizers.Optimizer):
                    opts.append((_name_for(v, i), v))
        else:
            if isinstance(opt_obj, keras.optimizers.Optimizer):
                opts.append((_name_for(opt_obj, 0), opt_obj))

        return opts

    def _current_lr(self, optimizer: keras.optimizers.Optimizer) -> float:
        """
        Try hard to obtain the current (decayed) LR for this optimizer.
        Order:
          1) If LR is schedule/callable → call with current global step.
          2) Try private `_decayed_lr(dtype)` (covers many TF2.x optimizers).
          3) Read optimizer.learning_rate / optimizer.lr as tensor/variable/float.
        """
        # Current global step/iterations
        step = getattr(optimizer, "iterations", None)
        if isinstance(step, (tf.Variable, tf.Tensor)):
            step_val = step  # pass tensor to schedule to keep graph-compat where needed
        else:
            # fallback to 0
            step_val = tf.convert_to_tensor(0, dtype=tf.int64)

        # 1) learning_rate attribute
        lr_attr = getattr(optimizer, "learning_rate", None)
        if lr_attr is None:
            lr_attr = getattr(optimizer, "lr", None)

        # 1.a) Schedule/callable
        if _is_schedule(lr_attr):
            try:
                val = lr_attr(step_val)  # type: ignore[operator]
                return _to_float(val)
            except Exception:
                pass

        # 2) Private decayed LR (OptimizerV2). Works for many TF2.x versions.
        decayed = getattr(optimizer, "_decayed_lr", None)
        if callable(decayed):
            try:
                val = decayed(self._dtype_for_decayed_lr)
                return _to_float(val)
            except Exception:
                pass

        # 3) Direct value (variable/tensor/float)
        if lr_attr is not None:
            return _to_float(lr_attr)

        # Final fallback: some very custom optimizers keep hyper dicts
        try:
            hypers = getattr(optimizer, "_hyper", None) or {}
            if "learning_rate" in hypers:
                return _to_float(hypers["learning_rate"])
        except Exception:
            pass

        # If all else fails, return NaN rather than crashing training.
        return float("nan")

    def _write_lrs_into_logs(self, logs: Optional[Dict[str, Any]]) -> None:
        log_dict: Dict[str, Any] = logs if logs is not None else {}
        if not self._optimizers:
            # Nothing to log if no optimizer detected
            log_dict[self.log_name] = float("nan")
            return

        if len(self._optimizers) == 1:
            _, opt = self._optimizers[0]
            log_dict[self.log_name] = self._current_lr(opt)
            return

        # Multiple optimizers: log one key per optimizer
        for name, opt in self._optimizers:
            log_dict[f"{self.log_name}/{name}"] = self._current_lr(opt)
