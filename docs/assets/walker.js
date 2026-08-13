// Nightingale — calibrated clinical risk models
// Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
// SPDX-License-Identifier: MIT

// The browser-side inference engine: an XGBoost tree-walker, a calibrator,
// and a conformal verdict, in plain ES2015 with zero dependencies.
//
// This is a statement-for-statement mirror of the reference walker in
// src/nightingale/export.py (`_walk_tree`, `_sigmoid`, `apply_calibrator`,
// `predict`). The two are kept trivially diffable on purpose:
// tests/test_parity.py scores 200 seeded cases per condition through both
// and requires the p_cal values to agree to 1e-9 and the conformal verdict
// to agree exactly. Any change here needs the same change there.
//
// Three details carry the parity:
//
//   1. Math.fround(x) < node.t — XGBoost compares feature values against
//      split thresholds in 32-bit float, and with tree_method="hist" the
//      thresholds ARE observed data values, so real rows land exactly on
//      them. Comparing in float64 sends those rows down the wrong branch.
//      Math.fround is IEEE-754 single rounding, identical to numpy.float32.
//      Thresholds arrive already float32-exact, so they need no rounding.
//
//   2. Leaf values are summed onto model.base_score in tree order, in
//      float64 — the same order Python uses, so the running sum is
//      bit-identical across the two languages.
//
//   3. model.base_score is the MARGIN offset (the logit of XGBoost's own
//      base_score probability, which is carried separately as
//      base_score_probability). It is precomputed at export time so V8's
//      Math.log and libm's log cannot disagree by a ulp here.
//
// Usage as a library (browser or node):
//     predict(model, x) -> { p_raw, p_cal, set }
// where `x` is an array positionally aligned to model.features, using
// null or NaN for anything not measured.
//
// Usage as a CLI:
//     node walker.js <model.json> <cases.json>
// prints a JSON array of { p_raw, p_cal, set } to stdout.

'use strict';

var LEAF = -1; // node.f sentinel: this node is a leaf, its value is node.v

// null (JSON's "not measured") and NaN both route down the missing branch.
function isMissing(value) {
  return value === null || value === undefined || (typeof value === 'number' && isNaN(value));
}

// The leaf value one flat tree assigns to x. Nodes reference each other by
// POSITION in the array, never by any id carried over from XGBoost's dump.
function walkTree(nodes, x) {
  var index = 0;
  while (nodes[index].f !== LEAF) {
    var node = nodes[index];
    var value = x[node.f];
    if (isMissing(value)) {
      index = node.m;
    } else if (Math.fround(value) < node.t) {
      index = node.l;
    } else {
      index = node.r;
    }
  }
  return nodes[index].v;
}

// 1 / (1 + exp(-z)), taking whichever algebraically identical branch
// cannot overflow. Python's _sigmoid branches the same way.
function sigmoid(z) {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z));
  }
  var e = Math.exp(z);
  return e / (1 + e);
}

// Apply an exported calibrator block to a raw probability.
//
// sigmoid: 1 / (1 + exp(a * p + b)) — applied to the probability itself,
// not its logit (this is the exact form nightingale.calibrate fits).
//
// isotonic: linear interpolation between ascending (x, y) breakpoints,
// clamped to the end values outside [x[0], x[-1]] — numpy.interp's
// semantics, including its slope * (p - x0) + y0 evaluation order, which
// is what makes this agree with the fitted Python calibrator to the ulp.
function applyCalibrator(calibrator, p) {
  if (calibrator.type === 'sigmoid') {
    return sigmoid(-(calibrator.a * p + calibrator.b));
  }
  if (calibrator.type === 'isotonic') {
    var xs = calibrator.x;
    var ys = calibrator.y;
    if (p <= xs[0]) {
      return ys[0];
    }
    if (p >= xs[xs.length - 1]) {
      return ys[ys.length - 1];
    }
    var lo = 0;
    var hi = xs.length - 1;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (xs[mid] <= p) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    if (xs[lo + 1] === xs[lo]) {
      return ys[lo + 1];
    }
    var slope = (ys[lo + 1] - ys[lo]) / (xs[lo + 1] - xs[lo]);
    return slope * (p - xs[lo]) + ys[lo];
  }
  throw new Error('unknown calibrator type ' + calibrator.type);
}

// Collapse a conformal prediction set to one of positive/negative/uncertain.
// Mirrors nightingale.conformal.prediction_set, including the deliberate
// argmax fallback for an EMPTY set: "neither label is plausible" is a
// different situation from "both are" (which is what "uncertain" means),
// and blurring the two together would hide a real failure mode.
function predictionSet(p, qHat) {
  var positiveIn = p >= 1 - qHat;
  var negativeIn = p <= qHat;
  if (positiveIn && negativeIn) {
    return 'uncertain';
  }
  if (positiveIn) {
    return 'positive';
  }
  if (negativeIn) {
    return 'negative';
  }
  return p >= 0.5 ? 'positive' : 'negative';
}

// Score one row: { p_raw, p_cal, set }.
function predict(model, x) {
  var total = model.base_score;
  for (var i = 0; i < model.trees.length; i++) {
    total += walkTree(model.trees[i], x);
  }
  var pRaw = sigmoid(total);
  var pCal = applyCalibrator(model.calibrator, pRaw);
  return {
    p_raw: pRaw,
    p_cal: pCal,
    set: predictionSet(pCal, model.conformal.q_hat)
  };
}

// Re-walk the model's own canary block and report the largest deviation
// from the stored p_cal. A browser can call this on load to prove the
// model.json it just downloaded is internally consistent — no server, no
// retraining, no trust required.
function checkCanaries(model) {
  var worst = 0;
  for (var i = 0; i < model.canaries.inputs.length; i++) {
    var diff = Math.abs(predict(model, model.canaries.inputs[i]).p_cal - model.canaries.p_cal[i]);
    if (diff > worst) {
      worst = diff;
    }
  }
  return { max_abs_diff: worst, tolerance: model.canaries.tolerance, ok: worst <= model.canaries.tolerance };
}

// CommonJS export for node (and the parity test); harmless in a browser,
// where the functions above are already globals of the loaded script.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    predict: predict,
    walkTree: walkTree,
    applyCalibrator: applyCalibrator,
    predictionSet: predictionSet,
    sigmoid: sigmoid,
    checkCanaries: checkCanaries
  };
}

// CLI: only when this file is the program being run, never when it is
// require()d as a library.
if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
  var fs = require('fs');
  var argv = process.argv.slice(2);
  if (argv.length !== 2) {
    process.stderr.write('usage: node walker.js <model.json> <cases.json>\n');
    process.exit(2);
  }
  var loadedModel = JSON.parse(fs.readFileSync(argv[0], 'utf8'));
  var cases = JSON.parse(fs.readFileSync(argv[1], 'utf8'));
  var results = cases.map(function (row) {
    return predict(loadedModel, row);
  });
  process.stdout.write(JSON.stringify(results) + '\n');
}
