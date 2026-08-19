// 自包含 DSP 工具：Bluestein FFT（任意 N）、STFT/ISTFT、Griffin-Lim 声码器、WAV 写出。
// 无第三方依赖（不依赖 FFTW/ffmpeg/torch），仅标准库。
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace dsp {

constexpr int kSr = 24000;
constexpr int kNfft = 1920;
constexpr int kHop = 480;
constexpr int kNBins = kNfft / 2 + 1;  // 961
constexpr int kNMels = 80;

using Cx = std::complex<float>;

inline void FftRadix2(std::vector<Cx>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        // twiddle 用 double 递推（float32 递推在 j 大时累积到 ~1e-3，不可接受）
        const double ang = (inverse ? 2.0 : -2.0) * M_PI / len;
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                Cx u = a[i + j];
                Cx v = a[i + j + len / 2] *
                       Cx(static_cast<float>(w.real()), static_cast<float>(w.imag()));
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& x : a) x /= static_cast<float>(n);
    }
}

struct FftPlan {
    int n = 0;
    int m = 0;
    std::vector<Cx> chirp;
    std::vector<Cx> b;
};

inline FftPlan& GetPlan(int n) {
    static FftPlan p;
    if (p.n == n) return p;
    p.n = n;
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;
    p.m = m;
    p.chirp.assign(n, Cx(0, 0));
    for (int k = 0; k < n; ++k) {
        // 大 k 时 -πk²/N 相位很大，float32 三角函数会损失精度；用 double 计算并归约到 [0,2π)
        const double ph = std::fmod(-M_PI * static_cast<double>(k) * k / n, 2.0 * M_PI);
        p.chirp[k] = Cx(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
    }
    p.b.assign(m, Cx(0, 0));
    p.b[0] = std::conj(p.chirp[0]);
    for (int k = 1; k < n; ++k) {
        p.b[k] = std::conj(p.chirp[k]);
        p.b[m - k] = std::conj(p.chirp[k]);
    }
    FftRadix2(p.b, false);
    return p;
}

// 任意长度 N 的 FFT（Bluestein，内部 radix-2 大小 M = next_pow2(2N-1)）
inline std::vector<Cx> FftAny(const std::vector<Cx>& x) {
    const int n = static_cast<int>(x.size());
    FftPlan& p = GetPlan(n);
    std::vector<Cx> a(p.m, Cx(0, 0));
    for (int k = 0; k < n; ++k) a[k] = x[k] * p.chirp[k];
    FftRadix2(a, false);
    for (int k = 0; k < p.m; ++k) a[k] *= p.b[k];
    FftRadix2(a, true);
    std::vector<Cx> y(n);
    for (int k = 0; k < n; ++k) y[k] = a[k] * p.chirp[k];
    return y;
}

inline std::vector<Cx> IfftAny(const std::vector<Cx>& x) {
    std::vector<Cx> c(x.size());
    for (size_t k = 0; k < x.size(); ++k) c[k] = std::conj(x[k]);
    std::vector<Cx> y = FftAny(c);
    for (size_t k = 0; k < y.size(); ++k) {
        y[k] = std::conj(y[k]) / static_cast<float>(y.size());
    }
    return y;
}

inline std::vector<float> Hann() {
    std::vector<float> w(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (kNfft - 1)));
    }
    return w;
}

// center=false, hop=480, n_fft=1920 -> (961, T)
inline std::vector<std::vector<Cx>> Stft(const std::vector<float>& x, const std::vector<float>& win) {
    const int nframes = static_cast<int>((x.size() - kNfft) / kHop) + 1;
    std::vector<std::vector<Cx>> spec(kNBins, std::vector<Cx>(nframes, Cx(0, 0)));
    std::vector<Cx> frame(kNfft);
    for (int i = 0; i < nframes; ++i) {
        for (int j = 0; j < kNfft; ++j) frame[j] = Cx(x[i * kHop + j] * win[j], 0.0f);
        std::vector<Cx> f = FftAny(frame);
        for (int b = 0; b < kNBins; ++b) spec[b][i] = f[b];
    }
    return spec;
}

inline std::vector<float> Istft(const std::vector<std::vector<Cx>>& spec, const std::vector<float>& win) {
    const int nframes = static_cast<int>(spec[0].size());
    const int n = (nframes - 1) * kHop + kNfft;
    std::vector<float> out(n, 0.0f), wsum(n, 0.0f);
    std::vector<Cx> frame(kNfft, Cx(0, 0));
    for (int i = 0; i < nframes; ++i) {
        frame.assign(kNfft, Cx(0, 0));
        for (int b = 0; b < kNBins; ++b) frame[b] = spec[b][i];
        for (int b = 1; b < kNfft - kNBins + 1; ++b) {
            frame[kNfft - b] = std::conj(frame[b]);
        }
        std::vector<Cx> t = IfftAny(frame);
        const int start = i * kHop;
        for (int j = 0; j < kNfft; ++j) {
            out[start + j] += t[j].real() * win[j];
            wsum[start + j] += win[j] * win[j];
        }
    }
    for (int i = 0; i < n; ++i) out[i] /= (wsum[i] + 1e-8f);
    return out;
}

// log10 mel（kNMels x frames，行主序）-> 24kHz 波形（Griffin-Lim，纯标准库）
inline std::vector<float> GriffinLim(const std::vector<float>& mel_log10, int frames,
                                     const std::vector<float>& mel_inv,
                                     const std::vector<float>& win, int iters = 20) {
    const float ln10 = 2.302585093f;
    std::vector<float> M(kNMels * frames);
    for (int m = 0; m < kNMels; ++m) {
        for (int t = 0; t < frames; ++t) {
            M[m * frames + t] = std::exp(mel_log10[m * frames + t] * ln10);
        }
    }
    std::vector<float> V(kNBins * frames, 0.0f);
    for (int b = 0; b < kNBins; ++b) {
        for (int t = 0; t < frames; ++t) {
            float acc = 0.0f;
            for (int m = 0; m < kNMels; ++m) {
                acc += mel_inv[b * kNMels + m] * M[m * frames + t];
            }
            V[b * frames + t] = acc > 0.0f ? acc : 0.0f;
        }
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> ph(0.0f, 2.0f * static_cast<float>(M_PI));
    std::vector<std::vector<Cx>> spec(kNBins, std::vector<Cx>(frames));
    for (int b = 0; b < kNBins; ++b) {
        for (int t = 0; t < frames; ++t) {
            spec[b][t] = Cx(V[b * frames + t] * std::cos(ph(gen)),
                            V[b * frames + t] * std::sin(ph(gen)));
        }
    }
    std::vector<float> x;
    for (int it = 0; it < iters; ++it) {
        x = Istft(spec, win);
        std::vector<std::vector<Cx>> X = Stft(x, win);
        for (int b = 0; b < kNBins; ++b) {
            for (int t = 0; t < frames; ++t) {
                const float ang = std::arg(X[b][t]);
                spec[b][t] = Cx(V[b * frames + t] * std::cos(ang),
                                V[b * frames + t] * std::sin(ang));
            }
        }
    }
    x = Istft(spec, win);
    float peak = 1e-8f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    for (float& v : x) v /= peak;
    return x;
}

// 24kHz 16-bit mono WAV -> bytes（纯标准库）
inline void WriteWav(const std::vector<float>& x, std::vector<char>* out) {
    const int n = static_cast<int>(x.size());
    const int data_bytes = n * 2;
    out->resize(44 + data_bytes);
    auto put32 = [&](int off, uint32_t v) {
        (*out)[off] = static_cast<char>(v & 0xff);
        (*out)[off + 1] = static_cast<char>((v >> 8) & 0xff);
        (*out)[off + 2] = static_cast<char>((v >> 16) & 0xff);
        (*out)[off + 3] = static_cast<char>((v >> 24) & 0xff);
    };
    std::memcpy(out->data(), "RIFF", 4);
    put32(4, 36 + data_bytes);
    std::memcpy(out->data() + 8, "WAVEfmt ", 8);
    put32(16, 16);
    (*out)[20] = 1;
    (*out)[21] = 0;  // PCM
    (*out)[22] = 1;
    (*out)[23] = 0;  // mono
    put32(24, kSr);
    put32(28, kSr * 2);
    (*out)[32] = 2;
    (*out)[33] = 0;  // block align
    (*out)[34] = 16;
    (*out)[35] = 0;  // bits
    std::memcpy(out->data() + 36, "data", 4);
    put32(40, data_bytes);
    for (int i = 0; i < n; ++i) {
        int16_t p = static_cast<int16_t>(
            std::max(-32768.0f, std::min(32767.0f, x[i] * 32767.0f)));
        out->at(44 + 2 * i) = static_cast<char>(p & 0xff);
        out->at(45 + 2 * i) = static_cast<char>((p >> 8) & 0xff);
    }
}

}  // namespace dsp
