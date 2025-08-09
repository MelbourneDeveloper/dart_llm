import 'dart:math';
import 'dart:typed_data';
import 'dart:io';

class Tensor {
  final List<int> shape;
  final Float64List data;
  final Float64List grad;
  String op = '';
  List<Tensor> parents = [];
  void Function()? _backward;

  Tensor(this.shape, Float64List? d)
      : data = d ?? Float64List(shape.reduce((a, b) => a * b)),
        grad = Float64List(shape.reduce((a, b) => a * b));

  static Tensor zeros(List<int> shape) => Tensor(shape, null);

  static Tensor randn(List<int> shape, Random rng, double std) {
    final t = Tensor(shape, null);
    for (int i = 0; i < t.data.length; i++) {
      double u1 = rng.nextDouble().clamp(1e-12, 1 - 1e-12);
      double u2 = rng.nextDouble();
      double z = sqrt(-2.0 * log(u1)) * cos(2 * pi * u2);
      t.data[i] = z * std;
    }
    return t;
  }

  Tensor clone() {
    final t = Tensor(List<int>.from(shape), Float64List.fromList(data));
    return t;
  }

  int get size => data.length;

  static Tensor add(Tensor a, Tensor b) {
    final ash = a.shape, bsh = b.shape;
    final outShape = _broadcastShape(ash, bsh);
    final out = Tensor(outShape, null);
    final aStr = _strides(ash);
    final bStr = _strides(bsh);
    final oStr = _strides(outShape);
    final idx = List.filled(outShape.length, 0);
    for (int i = 0; i < out.size; i++) {
      _unravel(i, oStr, idx);
      final ai = _ravel(_minIdx(idx, ash), aStr);
      final bi = _ravel(_minIdx(idx, bsh), bStr);
      out.data[i] = a.data[ai] + b.data[bi];
    }
    out.op = 'add';
    out.parents = [a, b];
    out._backward = () {
      for (int i = 0; i < out.size; i++) {
        _unravel(i, oStr, idx);
        final ai = _ravel(_minIdx(idx, ash), aStr);
        final bi = _ravel(_minIdx(idx, bsh), bStr);
        a.grad[ai] += out.grad[i];
        b.grad[bi] += out.grad[i];
      }
    };
    return out;
  }

  static Tensor sub(Tensor a, Tensor b) {
    final negb = mulScalar(b, -1.0);
    return add(a, negb);
  }

  static Tensor mul(Tensor a, Tensor b) {
    final ash = a.shape, bsh = b.shape;
    final outShape = _broadcastShape(ash, bsh);
    final out = Tensor(outShape, null);
    final aStr = _strides(ash);
    final bStr = _strides(bsh);
    final oStr = _strides(outShape);
    final idx = List.filled(outShape.length, 0);
    for (int i = 0; i < out.size; i++) {
      _unravel(i, oStr, idx);
      final ai = _ravel(_minIdx(idx, ash), aStr);
      final bi = _ravel(_minIdx(idx, bsh), bStr);
      out.data[i] = a.data[ai] * b.data[bi];
    }
    out.op = 'mul';
    out.parents = [a, b];
    out._backward = () {
      for (int i = 0; i < out.size; i++) {
        _unravel(i, oStr, idx);
        final ai = _ravel(_minIdx(idx, ash), aStr);
        final bi = _ravel(_minIdx(idx, bsh), bStr);
        final g = out.grad[i];
        a.grad[ai] += b.data[bi] * g;
        b.grad[bi] += a.data[ai] * g;
      }
    };
    return out;
  }

  static Tensor mulScalar(Tensor a, double s) {
    final out = Tensor(a.shape, null);
    for (int i = 0; i < a.size; i++) out.data[i] = a.data[i] * s;
    out.op = 'mulScalar';
    out.parents = [a];
    out._backward = () {
      for (int i = 0; i < a.size; i++) a.grad[i] += s * out.grad[i];
    };
    return out;
  }

  static Tensor addScalar(Tensor a, double s) {
    final out = Tensor(a.shape, null);
    for (int i = 0; i < a.size; i++) out.data[i] = a.data[i] + s;
    out.op = 'addScalar';
    out.parents = [a];
    out._backward = () {
      for (int i = 0; i < a.size; i++) a.grad[i] += out.grad[i];
    };
    return out;
  }

  static Tensor matmul(Tensor a, Tensor b) {
    final aSh = a.shape, bSh = b.shape;
    if (aSh.length != 2 || bSh.length != 2) {
      throw Exception('matmul expects 2D tensors');
    }
    final m = aSh[0], k = aSh[1], n = bSh[1];
    if (k != bSh[0]) throw Exception('incompatible matmul shapes');
    final out = Tensor([m, n], null);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double sumv = 0.0;
        for (int t = 0; t < k; t++) {
          sumv += a.data[i * k + t] * b.data[t * n + j];
        }
        out.data[i * n + j] = sumv;
      }
    }
    out.op = 'matmul';
    out.parents = [a, b];
    out._backward = () {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          final go = out.grad[i * n + j];
          for (int t = 0; t < k; t++) {
            a.grad[i * k + t] += b.data[t * n + j] * go;
            b.grad[t * n + j] += a.data[i * k + t] * go;
          }
        }
      }
    };
    return out;
  }

  static Tensor reshape(Tensor a, List<int> newShape) {
    final out = Tensor(newShape, a.data);
    out.op = 'reshape';
    out.parents = [a];
    out._backward = () {
      for (int i = 0; i < a.size; i++) a.grad[i] += out.grad[i];
    };
    return out;
  }

  static Tensor transpose2D(Tensor a) {
    if (a.shape.length != 2) throw Exception('transpose2D expects 2D');
    final m = a.shape[0], n = a.shape[1];
    final out = Tensor([n, m], null);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        out.data[j * m + i] = a.data[i * n + j];
      }
    }
    out.op = 'transpose2D';
    out.parents = [a];
    out._backward = () {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          a.grad[i * n + j] += out.grad[j * m + i];
        }
      }
    };
    return out;
  }

  static Tensor sum(Tensor a, {List<int>? axes, bool keepDims = false}) {
    final dims = a.shape.length;
    final ax = axes ?? List.generate(dims, (i) => i);
    final keep = List<bool>.filled(dims, false);
    for (final i in ax) keep[i] = true;
    final outShape = <int>[];
    for (int i = 0; i < dims; i++) {
      if (keep[i]) {
        if (keepDims) {
          outShape.add(1);
        }
      } else {
        outShape.add(a.shape[i]);
      }
    }
    if (outShape.isEmpty) {
      final out = Tensor([1], null);
      double s = 0.0;
      for (int i = 0; i < a.size; i++) s += a.data[i];
      out.data[0] = s;
      out.op = 'sum_all';
      out.parents = [a];
      out._backward = () {
        final g = out.grad[0];
        for (int i = 0; i < a.size; i++) a.grad[i] += g;
      };
      return out;
    }
    final out = Tensor(outShape, null);
    final aStr = _strides(a.shape);
    final oStr = _strides(outShape);
    final aIdx = List.filled(a.shape.length, 0);
    final oIdx = List.filled(outShape.length, 0);
    for (int ai = 0; ai < a.size; ai++) {
      _unravel(ai, aStr, aIdx);
      int pos = 0;
      for (int i = 0; i < dims; i++) {
        if (!keep[i]) {
          oIdx[pos] = aIdx[i];
          pos++;
        }
      }
      final oi = _ravel(oIdx, oStr);
      out.data[oi] += a.data[ai];
    }
    out.op = 'sum';
    out.parents = [a];
    out._backward = () {
      for (int ai = 0; ai < a.size; ai++) {
        _unravel(ai, aStr, aIdx);
        int pos = 0;
        for (int i = 0; i < dims; i++) {
          if (!keep[i]) {
            oIdx[pos] = aIdx[i];
            pos++;
          }
        }
        final oi = _ravel(oIdx, oStr);
        a.grad[ai] += out.grad[oi];
      }
    };
    return out;
  }

  static Tensor pow2(Tensor a) {
    final out = Tensor(a.shape, null);
    for (int i = 0; i < a.size; i++) out.data[i] = a.data[i] * a.data[i];
    out.op = 'pow2';
    out.parents = [a];
    out._backward = () {
      for (int i = 0; i < a.size; i++) a.grad[i] += 2.0 * a.data[i] * out.grad[i];
    };
    return out;
  }

static Tensor gelu(Tensor a) {
  final out = Tensor(a.shape, null);
  for (int i = 0; i < a.size; i++) {
    final x = a.data[i];
    final x3 = x * x * x;
    out.data[i] = 0.5 * x * (1.0 + _tanh(0.7978845608028654 * (x + 0.044715 * x3)));
  }
  out.op = 'gelu';
  out.parents = [a];
  out._backward = () {
    for (int i = 0; i < a.size; i++) {
      final x = a.data[i];
      final x2 = x * x;
      final x3 = x2 * x;
      final u = 0.7978845608028654 * (x + 0.044715 * x3);
      final t = _tanh(u);
      final sech2 = 1 - t * t; // d/dx tanh(u) = sech^2(u)
      final du = 0.7978845608028654 * (1 + 3 * 0.044715 * x2);
      final dgelu = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du;
      a.grad[i] += dgelu * out.grad[i];
    }
  };
  return out;
}


  static Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, double eps, int axis) {
    final nDims = x.shape.length;
    final axisSize = x.shape[axis];
    final outer = x.size ~/ axisSize;
    final out = Tensor(x.shape, null);
    final mean = Float64List(outer);
    final varr = Float64List(outer);

    for (int o = 0; o < outer; o++) {
      double m = 0.0;
      for (int i = 0; i < axisSize; i++) m += x.data[o * axisSize + i];
      m /= axisSize;
      mean[o] = m;
      double v = 0.0;
      for (int i = 0; i < axisSize; i++) {
        final d = x.data[o * axisSize + i] - m;
        v += d * d;
      }
      v /= axisSize;
      varr[o] = v;
      final invStd = 1.0 / sqrt(v + eps);
      for (int i = 0; i < axisSize; i++) {
        final idx = o * axisSize + i;
        final n = (x.data[idx] - m) * invStd;
        out.data[idx] = n * gamma.data[i] + beta.data[i];
      }
    }
    out.op = 'layernorm';
    out.parents = [x, gamma, beta];
    out._backward = () {
      final dx = Float64List(x.size);
      final dgamma = Float64List(gamma.size);
      final dbeta = Float64List(beta.size);
      for (int o = 0; o < outer; o++) {
        double m = mean[o];
        double v = varr[o];
        final invStd = 1.0 / sqrt(v + eps);
        double dmean = 0.0;
        double dvar = 0.0;
        for (int i = 0; i < axisSize; i++) {
          final idx = o * axisSize + i;
          final n = (x.data[idx] - m) * invStd;
          final go = out.grad[idx];
          dgamma[i] += go * n;
          dbeta[i] += go;
        }
        for (int i = 0; i < axisSize; i++) {
          final idx = o * axisSize + i;
          final n = (x.data[idx] - m) * invStd;
          final go = out.grad[idx];
          final dnorm = go * gamma.data[i];
          dvar += dnorm * (x.data[idx] - m) * (-0.5) * pow(v + eps, -1.5);
          dmean += -dnorm * invStd;
        }
        for (int i = 0; i < axisSize; i++) {
          final idx = o * axisSize + i;
          final dnorm = out.grad[idx] * gamma.data[i];
          dx[idx] += dnorm * invStd + dvar * 2.0 * (x.data[idx] - m) / axisSize + dmean / axisSize;
        }
      }
      for (int i = 0; i < x.size; i++) x.grad[i] += dx[i];
      for (int i = 0; i < gamma.size; i++) gamma.grad[i] += dgamma[i];
      for (int i = 0; i < beta.size; i++) beta.grad[i] += dbeta[i];
    };
    return out;
  }

  static Tensor softmaxLastDim(Tensor x) {
    final dims = x.shape;
    final N = dims.sublist(0, dims.length - 1).fold(1, (a, b) => a * b);
    final D = dims.last;
    final out = Tensor(x.shape, null);
    for (int n = 0; n < N; n++) {
      double maxv = -1e30;
      for (int d = 0; d < D; d++) {
        maxv = max(maxv, x.data[n * D + d]);
      }
      double sum = 0.0;
      for (int d = 0; d < D; d++) {
        final e = exp(x.data[n * D + d] - maxv);
        out.data[n * D + d] = e;
        sum += e;
      }
      for (int d = 0; d < D; d++) out.data[n * D + d] /= sum + 1e-12;
    }
    out.op = 'softmax';
    out.parents = [x];
    out._backward = () {
      for (int n = 0; n < N; n++) {
        double dot = 0.0;
        for (int d = 0; d < D; d++) {
          dot += out.data[n * D + d] * out.grad[n * D + d];
        }
        for (int d = 0; d < D; d++) {
          final yi = out.data[n * D + d];
          x.grad[n * D + d] += yi * (out.grad[n * D + d] - dot);
        }
      }
    };
    return out;
  }

  static Tensor concatLast(List<Tensor> xs) {
    if (xs.isEmpty) throw Exception('concat empty');
    final base = xs.first.shape.sublist(0, xs.first.shape.length - 1);
    int last = 0;
    for (final x in xs) {
      if (!_listEq(base, x.shape.sublist(0, x.shape.length - 1))) {
        throw Exception('concat shapes mismatch');
      }
      last += x.shape.last;
    }
    final outShape = [...base, last];
    final out = Tensor(outShape, null);
    int offset = 0;
    for (final x in xs) {
      final N = x.size;
      for (int i = 0; i < N; i++) {
        out.data[offset + i] = x.data[i];
      }
      x._concatOffset = offset;
      offset += N;
    }
    out.op = 'concat';
    out.parents = xs;
    out._backward = () {
      for (final x in xs) {
        for (int i = 0; i < x.size; i++) {
          x.grad[i] += out.grad[x._concatOffset + i];
        }
      }
    };
    return out;
  }

  int _concatOffset = 0;

  static List<int> _broadcastShape(List<int> a, List<int> b) {
    final n = max(a.length, b.length);
    final out = List<int>.filled(n, 0);
    for (int i = 0; i < n; i++) {
      final ai = i < n - a.length ? 1 : a[i - (n - a.length)];
      final bi = i < n - b.length ? 1 : b[i - (n - b.length)];
      if (ai != bi && ai != 1 && bi != 1) throw Exception('broadcast error');
      out[i] = max(ai, bi);
    }
    return out;
  }

  static List<int> _strides(List<int> shape) {
    final n = shape.length;
    final s = List<int>.filled(n, 0);
    int acc = 1;
    for (int i = n - 1; i >= 0; i--) {
      s[i] = acc;
      acc *= shape[i];
    }
    return s;
  }

  static void _unravel(int idx, List<int> strides, List<int> outIdx) {
    // Compute row-major multi-dimensional indices from a flat index using strides.
    // For strides computed as: strides[i] = product(shape[i+1..end]),
    // the index along dimension i is floor(idx / strides[i]) and then idx %= strides[i].
    for (int i = 0; i < strides.length; i++) {
      final s = strides[i];
      if (s == 0) {
        outIdx[i] = 0;
      } else {
        outIdx[i] = idx ~/ s;
        idx = idx % s;
      }
    }
  }

  static int _ravel(List<int> idx, List<int> strides) {
    int s = 0;
    for (int i = 0; i < idx.length; i++) s += idx[i] * strides[i];
    return s;
  }

  static List<int> _minIdx(List<int> idx, List<int> shape) {
    final n = idx.length, m = shape.length;
    final out = List<int>.filled(m, 0);
    for (int i = 0; i < m; i++) {
      final j = n - m + i;
      final v = j >= 0 ? idx[j] : 0;
      out[i] = shape[i] == 1 ? 0 : v;
    }
    return out;
  }

  static bool _listEq(List a, List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static void backward(Tensor loss) {
    final visited = <Tensor>{};
    final topo = <Tensor>[];
    void build(Tensor t) {
      if (visited.contains(t)) return;
      visited.add(t);
      for (final p in t.parents) build(p);
      topo.add(t);
    }

    build(loss);
    for (final t in topo) {
      for (int i = 0; i < t.grad.length; i++) t.grad[i] = 0.0;
    }
    loss.grad[0] = 1.0;
    for (int i = topo.length - 1; i >= 0; i--) {
      final t = topo[i];
      if (t._backward != null) t._backward!();
    }
  }
}

class Embedding {
  final Tensor weight;
  Embedding(int vocab, int dim, Random rng)
      : weight = Tensor.randn([vocab, dim], rng, 0.02);

  Tensor forward(Int32List idx) {
    final B = idx.length;
    final D = weight.shape[1];
    final out = Tensor([B, D], null);
    for (int i = 0; i < B; i++) {
      final row = idx[i];
      for (int d = 0; d < D; d++) {
        out.data[i * D + d] = weight.data[row * D + d];
      }
    }
    out.op = 'embedding';
    out.parents = [weight];
    out._backward = () {
      for (int i = 0; i < B; i++) {
        final row = idx[i];
        for (int d = 0; d < D; d++) {
          weight.grad[row * D + d] += out.grad[i * D + d];
        }
      }
    };
    return out;
  }
}

class Linear {
  final Tensor w;
  final Tensor b;
  Linear(int inF, int outF, Random rng, double stdScale)
      : w = Tensor.randn([inF, outF], rng, stdScale),
        b = Tensor.zeros([outF]);

  Tensor forward(Tensor x) {
    final y = Tensor.matmul(x, w);
    final sb = Tensor.reshape(b, [1, b.shape[0]]);
    final out = Tensor.add(y, sb);
    out.op = 'linear';
    out.parents = [y, sb];
    out._backward = () {
      for (int i = 0; i < y.size; i++) y.grad[i] += out.grad[i];
      for (int i = 0; i < sb.size; i++) b.grad[i] += out.grad[i];
    };
    return out;
  }
}

class MultiHeadSelfAttention {
  final int nHeads;
  final int nEmbed;
  final int headDim;
  final Linear qkv;
  final Linear proj;
  final int blockSize;
  final Tensor _mask;

  MultiHeadSelfAttention(this.nHeads, this.nEmbed, this.blockSize, Random rng)
      : headDim = nEmbed ~/ nHeads,
        qkv = Linear(nEmbed, 3 * nEmbed, rng, 0.02),
        proj = Linear(nEmbed, nEmbed, rng, 0.02),
        _mask = _buildMask(blockSize) {}

  static Tensor _buildMask(int T) {
    final t = Tensor.zeros([T, T]);
    for (int i = 0; i < T; i++) {
      for (int j = 0; j < T; j++) {
        t.data[i * T + j] = j <= i ? 0.0 : -1e9;
      }
    }
    return t;
  }

  Tensor forward(Tensor x, int B, int T) {
    final xt = Tensor.reshape(x, [B * T, nEmbed]);
    final qkvOut = qkv.forward(xt);
    final qkv3 = Tensor.reshape(qkvOut, [B, T, 3 * nEmbed]);
    final q = _sliceLast(qkv3, 0, nEmbed);
    final k = _sliceLast(qkv3, nEmbed, 2 * nEmbed);
    final v = _sliceLast(qkv3, 2 * nEmbed, 3 * nEmbed);
    final qh = _splitHeads(q, B, T);
    final kh = _splitHeads(k, B, T);
    final vh = _splitHeads(v, B, T);

    final scale = 1.0 / sqrt(headDim.toDouble());
    final scores = _attnScores(qh, kh, B, T, scale);
    final masked = Tensor.add(scores, _broadcastMask(B, nHeads, T));
    final probs = Tensor.softmaxLastDim(masked);
    final ctx = _attnApply(probs, vh, B, T);
    final merged = _mergeHeads(ctx, B, T);
    final y = Tensor.reshape(merged, [B * T, nEmbed]);
    final out = proj.forward(y);
    return out;
  }

  Tensor _sliceLast(Tensor x, int start, int end) {
    final B = x.shape[0], T = x.shape[1], C = x.shape[2];
    final W = end - start;
    final out = Tensor([B, T, W], null);
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int w = 0; w < W; w++) {
          out.data[b * T * W + t * W + w] = x.data[b * T * C + t * C + (start + w)];
        }
      }
    }
    out.op = 'slice';
    out.parents = [x];
    out._backward = () {
      for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
          for (int w = 0; w < W; w++) {
            x.grad[b * T * C + t * C + (start + w)] += out.grad[b * T * W + t * W + w];
          }
        }
      }
    };
    return out;
  }

  Tensor _splitHeads(Tensor x, int B, int T) {
    final out = Tensor([B, nHeads, T, headDim], null);
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int h = 0; h < nHeads; h++) {
          for (int d = 0; d < headDim; d++) {
            out.data[b * (nHeads * T * headDim) + h * (T * headDim) + t * headDim + d] =
                x.data[b * (T * nEmbed) + t * nEmbed + h * headDim + d];
          }
        }
      }
    }
    out.op = 'splitheads';
    out.parents = [x];
    out._backward = () {
      for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
          for (int h = 0; h < nHeads; h++) {
            for (int d = 0; d < headDim; d++) {
              x.grad[b * (T * nEmbed) + t * nEmbed + h * headDim + d] +=
                  out.grad[b * (nHeads * T * headDim) + h * (T * headDim) + t * headDim + d];
            }
          }
        }
      }
    };
    return out;
  }

  Tensor _mergeHeads(Tensor x, int B, int T) {
    final out = Tensor([B, T, nEmbed], null);
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int h = 0; h < nHeads; h++) {
          for (int d = 0; d < headDim; d++) {
            out.data[b * (T * nEmbed) + t * nEmbed + h * headDim + d] =
                x.data[b * (nHeads * T * headDim) + h * (T * headDim) + t * headDim + d];
          }
        }
      }
    }
    out.op = 'mergeheads';
    out.parents = [x];
    out._backward = () {
      for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
          for (int h = 0; h < nHeads; h++) {
            for (int d = 0; d < headDim; d++) {
              x.grad[b * (nHeads * T * headDim) + h * (T * headDim) + t * headDim + d] +=
                  out.grad[b * (T * nEmbed) + t * nEmbed + h * headDim + d];
            }
          }
        }
      }
    };
    return out;
  }

  Tensor _attnScores(Tensor q, Tensor k, int B, int T, double scale) {
    final out = Tensor([B, nHeads, T, T], null);
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < nHeads; h++) {
        for (int i = 0; i < T; i++) {
          for (int j = 0; j < T; j++) {
            double s = 0.0;
            for (int d = 0; d < headDim; d++) {
              final qi = q.data[b * (nHeads * T * headDim) + h * (T * headDim) + i * headDim + d];
              final kj = k.data[b * (nHeads * T * headDim) + h * (T * headDim) + j * headDim + d];
              s += qi * kj;
            }
            out.data[b * (nHeads * T * T) + h * (T * T) + i * T + j] = s * scale;
          }
        }
      }
    }
    out.op = 'scores';
    out.parents = [q, k];
    out._backward = () {
      for (int b = 0; b < B; b++) {
        for (int h = 0; h < nHeads; h++) {
          for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
              final g = out.grad[b * (nHeads * T * T) + h * (T * T) + i * T + j] * scale;
              for (int d = 0; d < headDim; d++) {
                final qiIdx =
                    b * (nHeads * T * headDim) + h * (T * headDim) + i * headDim + d;
                final kjIdx =
                    b * (nHeads * T * headDim) + h * (T * headDim) + j * headDim + d;
                final qi = q.data[qiIdx];
                final kj = k.data[kjIdx];
                q.grad[qiIdx] += kj * g;
                k.grad[kjIdx] += qi * g;
              }
            }
          }
        }
      }
    };
    return out;
  }

  Tensor _attnApply(Tensor p, Tensor v, int B, int T) {
    final out = Tensor([B, nHeads, T, headDim], null);
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < nHeads; h++) {
        for (int i = 0; i < T; i++) {
          for (int d = 0; d < headDim; d++) {
            double s = 0.0;
            for (int j = 0; j < T; j++) {
              final pij = p.data[b * (nHeads * T * T) + h * (T * T) + i * T + j];
              final vj = v.data[b * (nHeads * T * headDim) + h * (T * headDim) + j * headDim + d];
              s += pij * vj;
            }
            out.data[b * (nHeads * T * headDim) + h * (T * headDim) + i * headDim + d] = s;
          }
        }
      }
    }
    out.op = 'attnapply';
    out.parents = [p, v];
    out._backward = () {
      for (int b = 0; b < B; b++) {
        for (int h = 0; h < nHeads; h++) {
          for (int i = 0; i < T; i++) {
            for (int d = 0; d < headDim; d++) {
              final go = out.grad[b * (nHeads * T * headDim) + h * (T * headDim) + i * headDim + d];
              for (int j = 0; j < T; j++) {
                final pijIdx = b * (nHeads * T * T) + h * (T * T) + i * T + j;
                final vjIdx =
                    b * (nHeads * T * headDim) + h * (T * headDim) + j * headDim + d;
                final pij = p.data[pijIdx];
                final vj = v.data[vjIdx];
                p.grad[pijIdx] += vj * go;
                v.grad[vjIdx] += pij * go;
              }
            }
          }
        }
      }
    };
    return out;
  }

  Tensor _broadcastMask(int B, int H, int T) {
    final out = Tensor([B, H, T, T], null);
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) {
          for (int j = 0; j < T; j++) {
            out.data[b * (H * T * T) + h * (T * T) + i * T + j] =
                _mask.data[i * T + j];
          }
        }
      }
    }
    return out;
  }
}

class MLP {
  final Linear fc1;
  final Linear fc2;
  MLP(int nEmbed, int hiddenMult, Random rng)
      : fc1 = Linear(nEmbed, hiddenMult * nEmbed, rng, 0.02),
        fc2 = Linear(hiddenMult * nEmbed, nEmbed, rng, 0.02);

  Tensor forward(Tensor x, int B, int T, int C) {
    final xt = Tensor.reshape(x, [B * T, C]);
    final h1 = fc1.forward(xt);
    final h2 = Tensor.gelu(h1);
    final y = fc2.forward(h2);
    return y;
  }
}

class Block {
  final int nEmbed;
  final MultiHeadSelfAttention attn;
  final MLP mlp;
  final Tensor ln1g;
  final Tensor ln1b;
  final Tensor ln2g;
  final Tensor ln2b;

  Block(int nEmbed, int nHeads, int blockSize, Random rng)
      : nEmbed = nEmbed,
        attn = MultiHeadSelfAttention(nHeads, nEmbed, blockSize, rng),
        mlp = MLP(nEmbed, 4, rng),
        ln1g = Tensor.randn([nEmbed], rng, 0.02),
        ln1b = Tensor.zeros([nEmbed]),
        ln2g = Tensor.randn([nEmbed], rng, 0.02),
        ln2b = Tensor.zeros([nEmbed]);

  Tensor forward(Tensor x, int B, int T) {
    final xRes = x;
    final xNorm = Tensor.layerNorm(x, ln1g, ln1b, 1e-5, 2);
    final attnOut = attn.forward(xNorm, B, T);
    final attnRes = Tensor.add(Tensor.reshape(attnOut, [B, T, nEmbed]), xRes);
    final yNorm = Tensor.layerNorm(attnRes, ln2g, ln2b, 1e-5, 2);
    final mlpOut = mlp.forward(yNorm, B, T, nEmbed);
    final out = Tensor.add(Tensor.reshape(mlpOut, [B, T, nEmbed]), attnRes);
    return out;
  }
}

class GPT {
  final int vocabSize;
  final int blockSize;
  final int nEmbed;
  final int nLayer;
  final Embedding tokEmb;
  final Embedding posEmb;
  final List<Block> blocks;
  final Tensor lnFg;
  final Tensor lnFb;
  final Linear head;
  GPT(
      {required this.vocabSize,
      required this.blockSize,
      required this.nEmbed,
      required int nHead,
      required this.nLayer,
      required Random rng})
      : tokEmb = Embedding(vocabSize, nEmbed, rng),
        posEmb = Embedding(blockSize, nEmbed, rng),
        blocks = List.generate(
            nLayer, (_) => Block(nEmbed, nHead, blockSize, rng)),
        lnFg = Tensor.randn([nEmbed], rng, 0.02),
        lnFb = Tensor.zeros([nEmbed]),
        head = Linear(nEmbed, vocabSize, rng, 0.02);

  Tensor forward(Int32List idx) {
    final B = 1;
    final T = idx.length;
    final tok = tokEmb.forward(idx);
    final positions = Int32List(T);
    for (int i = 0; i < T; i++) positions[i] = i;
    final pos = posEmb.forward(positions);
    var x = Tensor.add(tok, pos);
    x = Tensor.reshape(x, [B, T, nEmbed]);
    for (final b in blocks) {
      x = b.forward(x, B, T);
    }
    x = Tensor.layerNorm(x, lnFg, lnFb, 1e-5, 2);
    final xt = Tensor.reshape(x, [B * T, nEmbed]);
    final logits = head.forward(xt);
    return Tensor.reshape(logits, [B, T, vocabSize]);
  }

  Map<String, Tensor> parameters() {
    final ps = <Tensor>[];
    ps.add(tokEmb.weight);
    ps.add(posEmb.weight);
    for (final b in blocks) {
      ps.addAll([b.attn.qkv.w, b.attn.qkv.b, b.attn.proj.w, b.attn.proj.b]);
      ps.addAll([b.ln1g, b.ln1b, b.ln2g, b.ln2b]);
      ps.addAll([b.mlp.fc1.w, b.mlp.fc1.b, b.mlp.fc2.w, b.mlp.fc2.b]);
    }
    ps.addAll([lnFg, lnFb, head.w, head.b]);
    final m = <String, Tensor>{};
    int i = 0;
    for (final p in ps) {
      m['p$i'] = p;
      i++;
    }
    return m;
  }
}

class CrossEntropyLoss {
  final int vocab;
  CrossEntropyLoss(this.vocab);

  Tensor forward(Tensor logits, Int32List targets) {
    final B = logits.shape[0], T = logits.shape[1];
    final V = logits.shape[2];
    final flat = Tensor.reshape(logits, [B * T, V]);
    final probs = Tensor.softmaxLastDim(flat);
    final loss = Tensor.zeros([1]);
    for (int i = 0; i < B * T; i++) {
      final ti = targets[i];
      final p = probs.data[i * V + ti].clamp(1e-12, 1.0);
      loss.data[0] += -log(p);
    }
    loss.data[0] /= (B * T);
    loss.op = 'celoss';
    loss.parents = [probs];
    loss._backward = () {
      for (int i = 0; i < B * T; i++) {
        for (int v = 0; v < V; v++) {
          final g = probs.data[i * V + v] / (B * T);
          probs.grad[i * V + v] += g;
        }
        final ti = targets[i];
        probs.grad[i * V + ti] += -1.0 / (B * T);
      }
    };
    return loss;
  }
}

class AdamW {
  final Map<String, Tensor> params;
  final double lr;
  final double bet1;
  final double bet2;
  final double eps;
  final double weightDecay;
  final Map<String, Float64List> m = {};
  final Map<String, Float64List> v = {};
  int t = 0;

  AdamW(this.params, this.lr, this.weightDecay,
      {this.bet1 = 0.9, this.bet2 = 0.999, this.eps = 1e-8}) {
    params.forEach((k, p) {
      m[k] = Float64List(p.size);
      v[k] = Float64List(p.size);
    });
  }

  void step() {
    t += 1;
    params.forEach((k, p) {
      final mk = m[k]!;
      final vk = v[k]!;
      for (int i = 0; i < p.size; i++) {
        final g = p.grad[i] + weightDecay * p.data[i];
        mk[i] = bet1 * mk[i] + (1 - bet1) * g;
        vk[i] = bet2 * vk[i] + (1 - bet2) * g * g;
        final mhat = mk[i] / (1 - pow(bet1, t));
        final vhat = vk[i] / (1 - pow(bet2, t));
        p.data[i] -= lr * mhat / (sqrt(vhat) + eps);
      }
    });
  }

  void zeroGrad() {
    params.forEach((k, p) {
      for (int i = 0; i < p.size; i++) p.grad[i] = 0.0;
    });
  }
}

class ByteTokenizer {
  int vocabSize() => 256;

  Int32List encode(String s) {
    final bytes = s.codeUnits;
    final out = Int32List(bytes.length);
    for (int i = 0; i < bytes.length; i++) {
      out[i] = bytes[i] & 0xFF;
    }
    return out;
  }

  String decode(List<int> ids) {
    return String.fromCharCodes(ids);
  }
}

List<List<Int32List>> makeBatches(Int32List data, int block, int batch, Random rng) {
  if (data.length < 2) {
    throw StateError('Not enough data to create input/target pairs (length=${data.length}).');
  }
  final effBlock = min(block, data.length - 1);
  final xs = <Int32List>[];
  final ys = <Int32List>[];
  final maxStartExclusive = max(1, data.length - effBlock - 1);
  for (int b = 0; b < batch; b++) {
    final start = maxStartExclusive == 1 ? 0 : rng.nextInt(maxStartExclusive);
    final x = Int32List(effBlock);
    final y = Int32List(effBlock);
    for (int t = 0; t < effBlock; t++) {
      x[t] = data[start + t];
      y[t] = data[start + t + 1];
    }
    xs.add(x);
    ys.add(y);
  }
  return [xs, ys];
}

Tensor batchForward(GPT model, List<Int32List> xs) {
  final B = xs.length;
  final T = xs[0].length;
  final logitsList = <Tensor>[];
  for (final x in xs) {
    logitsList.add(model.forward(x));
  }
  final cat = Tensor.concatLast(logitsList);
  return Tensor.reshape(cat, [B, T, model.vocabSize]);
}

Tensor batchLoss(CrossEntropyLoss cel, Tensor logits, List<Int32List> ys) {
  final B = ys.length;
  final T = ys[0].length;
  final targets = Int32List(B * T);
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      targets[b * T + t] = ys[b][t];
    }
  }
  return cel.forward(logits, targets);
}

List<int> sample(GPT model, List<int> context, int steps, Random rng) {
  final block = model.blockSize;
  final out = List<int>.from(context);
  for (int s = 0; s < steps; s++) {
    final start = out.length > block ? out.length - block : 0;
    final window = Int32List.fromList(out.sublist(start));
    final logits = model.forward(window);
    final last = logits.shape[1] - 1;
    final V = logits.shape[2];
    final lastRow = Float64List(V);
    for (int v = 0; v < V; v++) lastRow[v] = logits.data[last * V + v];
    final probs = _softmax1D(lastRow);
    final nextId = _sampleFromProbs(probs, rng);
    out.add(nextId);
  }
  return out;
}

Float64List _softmax1D(Float64List x) {
  double maxv = -1e30;
  for (final v in x) maxv = max(maxv, v);
  double sum = 0.0;
  final out = Float64List(x.length);
  for (int i = 0; i < x.length; i++) {
    final e = exp(x[i] - maxv);
    out[i] = e;
    sum += e;
  }
  for (int i = 0; i < x.length; i++) out[i] /= sum + 1e-12;
  return out;
}

int _sampleFromProbs(Float64List p, Random rng) {
  double r = rng.nextDouble();
  double c = 0.0;
  for (int i = 0; i < p.length; i++) {
    c += p[i];
    if (r <= c) return i;
  }
  return p.length - 1;
}

void main(List<String> args) async {
  final rng = Random(42);
  final tok = ByteTokenizer();
  final text = _loadToyText();
  final data = tok.encode(text);
  final trainFrac = 0.95;
  final split = (data.length * trainFrac).floor();
  final train = Int32List.fromList(data.sublist(0, split));
  final val = Int32List.fromList(data.sublist(split));

  final blockSize = 64;
  final model = GPT(
      vocabSize: tok.vocabSize(),
      blockSize: blockSize,
      nEmbed: 128,
      nHead: 4,
      nLayer: 2,
      rng: rng);

  final params = model.parameters();
  final opt = AdamW(params, 3e-4, 0.01);
  final cel = CrossEntropyLoss(model.vocabSize);
  final batchSize = 8;
  final iters = 300;

  for (int it = 1; it <= iters; it++) {
    final batch = makeBatches(train, blockSize, batchSize, rng);
    final xs = batch[0], ys = batch[1];
    final logits = batchForward(model, xs);
    final loss = batchLoss(cel, logits, ys);
    Tensor.backward(loss);
    opt.step();
    opt.zeroGrad();

    if (it % 25 == 0 || it == 1) {
      final vbatch = makeBatches(val, blockSize, batchSize, rng);
      final vLogits = batchForward(model, vbatch[0]);
      final vLoss = batchLoss(cel, vLogits, vbatch[1]);
      print('iter $it train=${loss.data[0].toStringAsFixed(4)} val=${vLoss.data[0].toStringAsFixed(4)}');
    }
  }

  final prompt = 'The quick brown ';
  final ids = tok.encode(prompt).toList();
  final gen = sample(model, ids, 200, rng);
  final out = tok.decode(gen);
  stdout.writeln('\n=== Sample ===\n$out');
}

String _loadToyText() {
  final s = '''
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversation?” So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
''';
  return s;
}


double _tanh(double x) {
  final e2x = exp(2.0 * x);
  return (e2x - 1.0) / (e2x + 1.0);
}
