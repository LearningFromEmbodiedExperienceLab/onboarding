(function () {
  "use strict";

  var PYODIDE_VERSION = "0.26.4";
  var PYODIDE_CDN =
    "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";

  /** @type {Promise<any> | null} */
  var pyodidePromise = null;

  /**
   * @typedef {{ title: string, setup: string, starter: string, validate: string, hint?: string }} Exercise
   */

  /** @type {Record<string, Exercise>} */
  var EXERCISES = {
    "pairwise-broadcast": {
      title: "1. Pairwise root distances (broadcasting)",
      setup: [
        "import numpy as np",
        "n_env = 4",
        "x = np.arange(n_env * 3 * 3, dtype=float).reshape(n_env, 3, 3)",
        "roots = x[:, 0, :]  # (n_env, 3) — body-0 position per env",
      ].join("\n"),
      starter: [
        "# dist[i, j] = ||roots[i] - roots[j]|| — no Python loops.",
        "# Insert axes so every root pairs with every root.",
        "delta = ...  # (n_env, n_env, 3)",
        "dist = ...   # (n_env, n_env)",
      ].join("\n"),
      validate: [
        "expected_delta = roots[:, None, :] - roots[None, :, :]",
        "assert delta.shape == (n_env, n_env, 3), f'delta.shape={delta.shape}, want (4, 4, 3)'",
        "assert np.allclose(delta, expected_delta), 'delta values wrong — check axis insertion'",
        "assert dist.shape == (n_env, n_env), f'dist.shape={dist.shape}, want (4, 4)'",
        "expected_dist = np.linalg.norm(expected_delta, axis=-1)",
        "assert np.allclose(dist, expected_dist), 'dist values wrong — use norm along last axis'",
      ].join("\n"),
      hint:
        "roots[:, None, :] is (n_env, 1, 3); roots[None, :, :] is (1, n_env, 3). " +
        "Subtract, then np.linalg.norm(..., axis=-1).",
    },

    "homogeneous-world-write": {
      title: "2.1 Homogeneous zip — world positions and write-back",
      setup: [
        "import numpy as np",
        "n_env, n_body = 4, 3",
        "x = np.arange(n_env * n_body * 3, dtype=float).reshape(n_env, n_body, 3)",
        "origins = np.array(",
        "    [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]",
        ")",
        "env_ids = np.array([0, 2])",
        "body_ids = np.array([1, 0])",
        "shift = np.array([1.0, 2.0, 3.0])",
        "x_ref = x.copy()",
      ].join("\n"),
      starter: [
        "# One (env, body) pair per index k (homogeneous zip).",
        "# 1) Gather env-frame positions (K, 3)",
        "# 2) Add world origins with broadcasting",
        "# 3) Write back x[env_ids, body_ids] = world + shift",
        "pos = x[env_ids, body_ids]",
        "world = ...",
        "x[env_ids, body_ids] = ...",
      ].join("\n"),
      validate: [
        "pos_expected = x_ref[env_ids, body_ids]",
        "world_expected = pos_expected + origins[env_ids]",
        "target = world_expected + shift",
        "assert world.shape == (len(env_ids), 3), f'world.shape={world.shape}, want (2, 3)'",
        "assert np.allclose(world, world_expected), 'world positions wrong — add origins[env_ids]'",
        "assert np.allclose(x[env_ids, body_ids], target), 'write-back wrong — assign world + shift'",
        "assert np.allclose(x[0, 2], x_ref[0, 2]), 'unchosen entries changed'",
        "assert np.allclose(x[2, 1], x_ref[2, 1]), 'unchosen entries changed'",
      ].join("\n"),
      hint:
        "origins[env_ids] has shape (K, 3) and broadcasts with pos (K, 3). " +
        "Indexed assignment writes through into x.",
    },

    "heterogeneous-world-write": {
      title: "2.2 Heterogeneous gather — grid of bodies per env",
      setup: [
        "import numpy as np",
        "n_env, n_body = 4, 3",
        "x = np.arange(n_env * n_body * 3, dtype=float).reshape(n_env, n_body, 3)",
        "origins = np.array(",
        "    [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]",
        ")",
        "env_ids = np.array([0, 2])",
        "body_ids = np.array([[0, 2], [1, 0]])  # (K, B)",
        "shift = np.array([1.0, 0.0, 0.0])",
        "x_ref = x.copy()",
        "K, B = body_ids.shape",
      ].join("\n"),
      starter: [
        "# env 0 → bodies 0,2; env 2 → bodies 1,0. Output grid (K, B, 3).",
        "# Expand index axes with None, gather xyz, offset by origins, write back.",
        "e = env_ids[:, None, None]   # (K, 1, 1)",
        "b = body_ids[:, :, None]     # (K, B, 1)",
        "pos = ...                    # (K, B, 3)",
        "world = ...                  # (K, B, 3)",
        "x[e, b, :] = ...",
      ].join("\n"),
      validate: [
        "e = env_ids[:, None, None]",
        "b = body_ids[:, :, None]",
        "pos_expected = x_ref[e, b, :]",
        "world_expected = pos_expected + origins[env_ids][:, None, :]",
        "target = world_expected + shift",
        "assert pos.shape == (K, B, 3), f'pos.shape={pos.shape}, want ({K}, {B}, 3)'",
        "assert np.allclose(pos, pos_expected), 'pos gather wrong — check e, b, and : on xyz'",
        "assert np.allclose(world, world_expected), 'world wrong — origins[env_ids][:, None, :]'",
        "assert np.allclose(x[e, b, :], target), 'write-back wrong — assign world + shift'",
        "assert np.allclose(x[1], x_ref[1]), 'unchosen env rows changed'",
      ].join("\n"),
      hint:
        "x[e, b, :] with e (K,1,1) and b (K,B,1) yields (K,B,3). " +
        "origins[env_ids][:, None, :] is (K,1,3) and broadcasts over B.",
    },

    "sequence-sampling": {
      title: "3. Sequence sampling — N segments of length T",
      setup: [
        "import numpy as np",
        "L = 100",
        "flat = np.arange(L, dtype=float)",
        "starts = np.array([0, 10, 50, 77])",
        "T = 5",
        "N = len(starts)",
      ].join("\n"),
      starter: [
        "# segments[n, t] = flat[starts[n] + t] — vectorized, no loop.",
        "# Build a (N, T) integer index grid with broadcasting.",
        "idx = ...       # (N, T)",
        "segments = ...  # (N, T)",
      ].join("\n"),
      validate: [
        "assert idx.shape == (N, T), f'idx.shape={idx.shape}, want ({N}, {T})'",
        "expected_idx = starts[:, None] + np.arange(T)[None, :]",
        "assert np.array_equal(idx, expected_idx), 'idx grid wrong — starts[:, None] + np.arange(T)[None, :]'",
        "assert segments.shape == (N, T), f'segments.shape={segments.shape}, want ({N}, {T})'",
        "expected_segments = flat[expected_idx]",
        "assert np.allclose(segments, expected_segments), 'segments values wrong — index flat with idx'",
        "assert np.allclose(segments[2, 0], flat[50]), 'spot check: segment 2 should start at flat[50]'",
      ].join("\n"),
      hint:
        "starts[:, None] has shape (N, 1); np.arange(T)[None, :] has shape (1, T). " +
        "Their sum broadcasts to (N, T). Then segments = flat[idx].",
    },
  };

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var existing = document.querySelector('script[src="' + src + '"]');
      if (existing) {
        if (existing.getAttribute("data-loaded") === "true") {
          resolve();
          return;
        }
        existing.addEventListener("load", function () {
          resolve();
        });
        existing.addEventListener("error", reject);
        return;
      }
      var script = document.createElement("script");
      script.src = src;
      script.async = true;
      script.onload = function () {
        script.setAttribute("data-loaded", "true");
        resolve();
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  function getPyodide() {
    if (!pyodidePromise) {
      pyodidePromise = loadScript(PYODIDE_CDN + "pyodide.js").then(function () {
        // eslint-disable-next-line no-undef
        return loadPyodide({ indexURL: PYODIDE_CDN });
      });
    }
    return pyodidePromise;
  }

  /**
   * @param {string} exerciseId
   * @param {string} userCode
   * @returns {Promise<{ ok: boolean, message?: string }>}
   */
  async function runExercise(exerciseId, userCode) {
    var exercise = EXERCISES[exerciseId];
    if (!exercise) {
      return { ok: false, message: "Unknown exercise: " + exerciseId };
    }

    var pyodide = await getPyodide();
    await pyodide.loadPackage("numpy");

    var script = [
      exercise.setup,
      "",
      userCode,
      "",
      exercise.validate,
    ].join("\n");

    try {
      await pyodide.runPythonAsync(script);
      return { ok: true };
    } catch (err) {
      var message = err && err.message ? String(err.message) : String(err);
      message = message.replace(/^PythonError: /, "");
      return { ok: false, message: message };
    }
  }

  function setStatus(container, kind, text) {
    var status = container.querySelector(".exercise-status");
    if (!status) {
      return;
    }
    status.className = "exercise-status " + kind;
    status.textContent = text;
  }

  function setFeedback(container, text) {
    var feedback = container.querySelector(".exercise-feedback");
    if (!feedback) {
      return;
    }
    if (text) {
      feedback.hidden = false;
      feedback.textContent = text;
    } else {
      feedback.hidden = true;
      feedback.textContent = "";
    }
  }

  function wireExercise(container) {
    var exerciseId = container.getAttribute("data-exercise-id");
    if (!exerciseId || !EXERCISES[exerciseId]) {
      return;
    }

    var exercise = EXERCISES[exerciseId];
    var textarea = container.querySelector("textarea.exercise-code");
    var button = container.querySelector("button.exercise-check");

    if (!textarea || !button) {
      return;
    }

    if (!textarea.value.trim()) {
      textarea.value = exercise.starter;
    }

    var titleEl = container.querySelector(".exercise-title");
    if (titleEl && !titleEl.textContent.trim()) {
      titleEl.textContent = exercise.title;
    }

    var hintDetails = container.querySelector("details.exercise-hint");
    if (hintDetails && exercise.hint) {
      var hintBody = hintDetails.querySelector(".hint-body");
      if (hintBody && !hintBody.textContent.trim()) {
        hintBody.textContent = exercise.hint;
      }
    }

    button.addEventListener("click", function () {
      button.disabled = true;
      setStatus(container, "loading", "Loading Python (first check may take a few seconds)…");
      setFeedback(container, "");

      runExercise(exerciseId, textarea.value)
        .then(function (result) {
          if (result.ok) {
            setStatus(container, "pass", "Correct");
            setFeedback(container, "");
          } else {
            setStatus(container, "fail", "Not yet — see details below");
            setFeedback(container, result.message || "Check failed.");
          }
        })
        .catch(function (err) {
          setStatus(container, "fail", "Runtime error");
          setFeedback(container, String(err));
        })
        .finally(function () {
          button.disabled = false;
        });
    });
  }

  function init() {
    var nodes = document.querySelectorAll(".tensor-exercise");
    if (!nodes.length) {
      return;
    }
    nodes.forEach(wireExercise);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
