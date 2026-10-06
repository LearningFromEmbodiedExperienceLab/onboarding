(function () {
  "use strict";

  var PAD = 36;
  var CURVE_SAMPLES = 200;

  /** @typedef {[number, number]} Point */

  /**
   * @param {HTMLCanvasElement} canvas
   */
  function setupHiDpi(canvas) {
    var rect = canvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    var w = Math.max(1, Math.round(rect.width * dpr));
    var h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    var ctx = canvas.getContext("2d");
    if (!ctx) {
      return null;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /**
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   */
  function worldToPx(x, y, w, h) {
    return {
      x: PAD + x * (w - 2 * PAD),
      y: h - PAD - y * (h - 2 * PAD),
    };
  }

  /**
   * @param {number} px
   * @param {number} py
   * @param {number} w
   * @param {number} h
   * @returns {Point}
   */
  function pxToWorld(px, py, w, h) {
  return [
      Math.min(1, Math.max(0, (px - PAD) / (w - 2 * PAD))),
      Math.min(1, Math.max(0, (h - PAD - py) / (h - 2 * PAD))),
    ];
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   */
  function drawGrid(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = "#e6e9ee";
    ctx.lineWidth = 1;
    for (var i = 1; i < 5; i++) {
      var t = i / 5;
      var x = PAD + t * (w - 2 * PAD);
      var y = h - PAD - t * (h - 2 * PAD);
      ctx.beginPath();
      ctx.moveTo(x, PAD);
      ctx.lineTo(x, h - PAD);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PAD, y);
      ctx.lineTo(w - PAD, y);
      ctx.stroke();
    }
    ctx.strokeStyle = "#b4bac4";
    ctx.strokeRect(PAD, PAD, w - 2 * PAD, h - 2 * PAD);
    ctx.restore();
  }

  /**
   * @param {Point[]} points
   * @returns {Point[]}
   */
  function convexHull2d(points) {
    if (points.length <= 1) {
      return points.slice();
    }
    var pts = points
      .map(function (p, i) {
        return { x: p[0], y: p[1], i: i };
      })
      .sort(function (a, b) {
        return a.y === b.y ? a.x - b.x : a.y - b.y;
      });

    function cross(o, a, b) {
      return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    }

    var lower = [];
    for (var i = 0; i < pts.length; i++) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pts[i]) <= 0) {
        lower.pop();
      }
      lower.push(pts[i]);
    }
    var upper = [];
    for (var j = pts.length - 1; j >= 0; j--) {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pts[j]) <= 0) {
        upper.pop();
      }
      upper.push(pts[j]);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper).map(function (p) {
      return [p.x, p.y];
    });
  }

  /**
   * @param {Point} p0
   * @param {Point} p1
   * @param {Point} p2
   * @param {Point} p3
   * @param {number} u
   * @returns {Point}
   */
  function cubicBezierPoint(p0, p1, p2, p3, u) {
    var omu = 1 - u;
    var omu2 = omu * omu;
    var omu3 = omu2 * omu;
    var u2 = u * u;
    var u3 = u2 * u;
    return [
      omu3 * p0[0] + 3 * omu2 * u * p1[0] + 3 * omu * u2 * p2[0] + u3 * p3[0],
      omu3 * p0[1] + 3 * omu2 * u * p1[1] + 3 * omu * u2 * p2[1] + u3 * p3[1],
    ];
  }

  /**
   * Natural cubic spline through (t[i], y[i]); evaluate at xEval.
   * @param {number[]} t
   * @param {number[]} y
   * @param {number[]} xEval
   */
  function cubicSpline1D(t, y, xEval) {
    var n = t.length;
    if (n < 2) {
      return xEval.map(function () {
        return y[0] || 0;
      });
    }
    if (n === 2) {
      return xEval.map(function (x) {
        var a = (y[1] - y[0]) / (t[1] - t[0]);
        return y[0] + a * (x - t[0]);
      });
    }

    var h = new Array(n - 1);
    for (var i = 0; i < n - 1; i++) {
      h[i] = t[i + 1] - t[i];
    }

    var alpha = new Array(n);
    alpha[0] = 0;
    alpha[n - 1] = 0;
    for (var j = 1; j < n - 1; j++) {
      alpha[j] =
        (3 / h[j]) * (y[j + 1] - y[j]) - (3 / h[j - 1]) * (y[j] - y[j - 1]);
    }

    var l = new Array(n);
    var mu = new Array(n);
    var z = new Array(n);
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;
    for (var k = 1; k < n - 1; k++) {
      l[k] = 2 * (t[k + 1] - t[k - 1]) - h[k - 1] * mu[k - 1];
      mu[k] = h[k] / l[k];
      z[k] = (alpha[k] - h[k - 1] * z[k - 1]) / l[k];
    }
    l[n - 1] = 1;
    z[n - 1] = 0;

    var c = new Array(n);
    var b = new Array(n - 1);
    var d = new Array(n - 1);
    c[n - 1] = 0;
    for (var m = n - 2; m >= 0; m--) {
      c[m] = z[m] - mu[m] * c[m + 1];
      b[m] = (y[m + 1] - y[m]) / h[m] - (h[m] * (c[m + 1] + 2 * c[m])) / 3;
      d[m] = (c[m + 1] - c[m]) / (3 * h[m]);
    }

    return xEval.map(function (x) {
      if (x <= t[0]) {
        return y[0];
      }
      if (x >= t[n - 1]) {
        return y[n - 1];
      }
      var seg = 0;
      while (seg < n - 2 && x > t[seg + 1]) {
        seg++;
      }
      var dx = x - t[seg];
      return y[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
    });
  }

  /**
   * @param {Point[]} waypoints
   * @param {number} samples
   * @returns {Point[]}
   */
  function interpolatingSpline2D(waypoints, samples) {
    var n = waypoints.length;
    var t = [];
    for (var i = 0; i < n; i++) {
      t.push(n === 1 ? 0 : i / (n - 1));
    }
    var uEval = [];
    for (var s = 0; s < samples; s++) {
      uEval.push(s / (samples - 1));
    }
    var xs = waypoints.map(function (p) {
      return p[0];
    });
    var ys = waypoints.map(function (p) {
      return p[1];
    });
    var xOut = cubicSpline1D(t, xs, uEval);
    var yOut = cubicSpline1D(t, ys, uEval);
    var curve = [];
    for (var k = 0; k < samples; k++) {
      curve.push([xOut[k], yOut[k]]);
    }
    return curve;
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {Point[]} pts
   * @param {number} w
   * @param {number} h
   * @param {{ color?: string, width?: number, closed?: boolean }} opts
   */
  function strokePolyline(ctx, pts, w, h, opts) {
    if (pts.length < 2) {
      return;
    }
    ctx.save();
    ctx.strokeStyle = opts.color || "#185fbf";
    ctx.lineWidth = opts.width || 2;
    ctx.beginPath();
    var p0 = worldToPx(pts[0][0], pts[0][1], w, h);
    ctx.moveTo(p0.x, p0.y);
    for (var i = 1; i < pts.length; i++) {
      var p = worldToPx(pts[i][0], pts[i][1], w, h);
      ctx.lineTo(p.x, p.y);
    }
    if (opts.closed) {
      ctx.closePath();
    }
    ctx.stroke();
    ctx.restore();
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {Point[]} hull
   * @param {number} w
   * @param {number} h
   */
  function fillHull(ctx, hull, w, h) {
    if (hull.length < 3) {
      return;
    }
    ctx.save();
    ctx.fillStyle = "rgba(255, 200, 185, 0.35)";
    ctx.strokeStyle = "rgba(255, 150, 120, 0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    var hp0 = worldToPx(hull[0][0], hull[0][1], w, h);
    ctx.moveTo(hp0.x, hp0.y);
    for (var i = 1; i < hull.length; i++) {
      var hp = worldToPx(hull[i][0], hull[i][1], w, h);
      ctx.lineTo(hp.x, hp.y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {Point} pt
   * @param {string} label
   * @param {number} w
   * @param {number} h
   */
  function drawHandle(ctx, pt, label, w, h) {
    var px = worldToPx(pt[0], pt[1], w, h);
    ctx.save();
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#185fbf";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px.x, px.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#283241";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(label, px.x + 9, px.y - 6);
    ctx.restore();
  }

  /**
   * @param {HTMLElement} root
   */
  function initBezierDemo(root) {
    /** @type {Point[]} */
    var controls = [
      [0.08, 0.12],
      [0.22, 0.78],
      [0.62, 0.68],
      [0.88, 0.15],
    ];
    var labels = ["P0", "P1", "P2", "P3"];
    var canvas = root.querySelector("canvas");
    if (!canvas) {
      return;
    }

    /** @type {number | null} */
    var dragIdx = null;

    function draw() {
      var rect = canvas.getBoundingClientRect();
      var w = rect.width;
      var h = rect.height;
      var ctx = setupHiDpi(canvas);
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, w, h);
      drawGrid(ctx, w, h);

      var hull = convexHull2d(controls);
      fillHull(ctx, hull, w, h);
      strokePolyline(ctx, controls, w, h, { color: "#7896c8", width: 2 });

      var curve = [];
      for (var i = 0; i < CURVE_SAMPLES; i++) {
        var u = i / (CURVE_SAMPLES - 1);
        curve.push(cubicBezierPoint(controls[0], controls[1], controls[2], controls[3], u));
      }
      strokePolyline(ctx, curve, w, h, { color: "#185fbf", width: 3 });

      for (var j = 0; j < controls.length; j++) {
        drawHandle(ctx, controls[j], labels[j], w, h);
      }
    }

    function hitTest(clientX, clientY) {
      var rect = canvas.getBoundingClientRect();
      var w = rect.width;
      var h = rect.height;
      for (var i = controls.length - 1; i >= 0; i--) {
        var px = worldToPx(controls[i][0], controls[i][1], w, h);
        var dx = clientX - rect.left - px.x;
        var dy = clientY - rect.top - px.y;
        if (dx * dx + dy * dy <= 12 * 12) {
          return i;
        }
      }
      return null;
    }

    function onPointerDown(ev) {
      var idx = hitTest(ev.clientX, ev.clientY);
      if (idx === null) {
        return;
      }
      dragIdx = idx;
      canvas.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    }

    function onPointerMove(ev) {
      if (dragIdx === null) {
        return;
      }
      var rect = canvas.getBoundingClientRect();
      controls[dragIdx] = pxToWorld(ev.clientX - rect.left, ev.clientY - rect.top, rect.width, rect.height);
      draw();
      ev.preventDefault();
    }

    function onPointerUp(ev) {
      if (dragIdx !== null) {
        canvas.releasePointerCapture(ev.pointerId);
        dragIdx = null;
      }
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    window.addEventListener("resize", draw);
    draw();
  }

  /**
   * @param {HTMLElement} root
   */
  function initBsplineDemo(root) {
    /** @type {Point[]} */
    var waypoints = [
      [0.1, 0.18],
      [0.24, 0.72],
      [0.42, 0.55],
      [0.64, 0.22],
      [0.86, 0.62],
    ];
    var canvas = root.querySelector("canvas");
    if (!canvas) {
      return;
    }

    /** @type {number | null} */
    var dragIdx = null;

    function draw() {
      var rect = canvas.getBoundingClientRect();
      var w = rect.width;
      var h = rect.height;
      var ctx = setupHiDpi(canvas);
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, w, h);
      drawGrid(ctx, w, h);

      strokePolyline(ctx, waypoints, w, h, { color: "#a0a8b8", width: 1 });

      var curve = interpolatingSpline2D(waypoints, CURVE_SAMPLES);
      strokePolyline(ctx, curve, w, h, { color: "#185fbf", width: 3 });

      for (var j = 0; j < waypoints.length; j++) {
        drawHandle(ctx, waypoints[j], "W" + j, w, h);
      }
    }

    function hitTest(clientX, clientY) {
      var rect = canvas.getBoundingClientRect();
      var w = rect.width;
      var h = rect.height;
      for (var i = waypoints.length - 1; i >= 0; i--) {
        var px = worldToPx(waypoints[i][0], waypoints[i][1], w, h);
        var dx = clientX - rect.left - px.x;
        var dy = clientY - rect.top - px.y;
        if (dx * dx + dy * dy <= 12 * 12) {
          return i;
        }
      }
      return null;
    }

    function onPointerDown(ev) {
      var idx = hitTest(ev.clientX, ev.clientY);
      if (idx === null) {
        return;
      }
      dragIdx = idx;
      canvas.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    }

    function onPointerMove(ev) {
      if (dragIdx === null) {
        return;
      }
      var rect = canvas.getBoundingClientRect();
      waypoints[dragIdx] = pxToWorld(
        ev.clientX - rect.left,
        ev.clientY - rect.top,
        rect.width,
        rect.height
      );
      draw();
      ev.preventDefault();
    }

    function onPointerUp(ev) {
      if (dragIdx !== null) {
        canvas.releasePointerCapture(ev.pointerId);
        dragIdx = null;
      }
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    window.addEventListener("resize", draw);
    draw();
  }

  function initAll() {
    document.querySelectorAll(".trajectory-spline-demo").forEach(function (root) {
      var kind = root.getAttribute("data-demo");
      if (kind === "bezier") {
        initBezierDemo(root);
      } else if (kind === "bspline") {
        initBsplineDemo(root);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
