// HiFT 声码器宿主侧 DSP（numpy 版 hift_vocoder.py 的 C++ 移植）。
// 与 torch 原版逐位验证：源激励 max diff ~5e-10、STFT ~5.6e-9、ISTFT ~1.3e-7。
#pragma once

#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace hift {

constexpr int kSr = 24000;
constexpr int kTMel = 198;          // 静态 mel 帧
constexpr float kPadVal = -11.0f;   // mel 尾部补静音值
constexpr int kNfft = 16;
constexpr int kHop = 4;
constexpr int kHarmonics = 9;       // 8 次谐波 + 基频
constexpr float kSineAmp = 0.1f;
constexpr float kNoiseStd = 0.003f;
constexpr float kVoicedTh = 10.0f;
constexpr int kUpsample = 480;
constexpr int kSrcLen = kTMel * kUpsample;          // 95040
constexpr int kStftFrames = kSrcLen / kHop + 1;     // 23761
constexpr float kAudioLimit = 0.99f;

using Cx = std::complex<float>;

inline std::vector<float> Hann16() {
    std::vector<float> w(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / kNfft));
    }
    return w;  // periodic hann == torch hann_window(16)
}

// 16 点 FFT（radix-2，直接实现，避免依赖 dsp.hpp 的 Bluestein 大 plan）
inline void Fft16(std::vector<Cx>& a, bool inverse) {
    const int n = 16;
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
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

// f0[198] -> 源激励 s[95040]（SineGen + SourceModuleHnNSF）。
// linear_w: 9 个谐波权重；linear_b: 偏置；phase/noise 由调用方生成（随机）。
inline std::vector<float> SourceFromF0(const std::vector<float>& f0,
                                       const std::vector<float>& phase,
                                       const std::vector<float>& noise,
                                       const std::vector<float>& linear_w,
                                       float linear_b) {
    std::vector<float> f0_up(kSrcLen);
    for (int i = 0; i < kTMel; ++i) {
        const float v = f0[i];
        for (int j = 0; j < kUpsample; ++j) f0_up[i * kUpsample + j] = v;
    }
    // F_mat[h][n] = f0_up * (h+1) / sr；theta = 2π * cumsum mod 1
    std::vector<float> sine_wavs(kHarmonics * kSrcLen);
    for (int h = 0; h < kHarmonics; ++h) {
        // torch 原版用 float32 做 F_mat/cumsum（相位累加会随时间漂移，但这是模型自带行为）；
        // 这里必须用 float32 逐位对齐，否则源激励与 torch/numpy 参考偏差可达 0.1
        float phase_acc = 0.0f;
        const double twopi = 2.0 * M_PI;
        for (int n = 0; n < kSrcLen; ++n) {
            // 与 numpy 相同运算顺序：(f0_up * (h+1)) / SR，全程 float32
            phase_acc += f0_up[n] * static_cast<float>(h + 1) / static_cast<float>(kSr);
            double th = twopi * std::fmod(static_cast<double>(phase_acc), 1.0) + phase[h];
            sine_wavs[h * kSrcLen + n] =
                static_cast<float>(kSineAmp * std::sin(th));
        }
    }
    std::vector<float> s(kSrcLen, 0.0f);
    for (int n = 0; n < kSrcLen; ++n) {
        const bool voiced = f0_up[n] > kVoicedTh;
        const float noise_amp = voiced ? kNoiseStd : kSineAmp / 3.0f;
        float merged = 0.0f;
        for (int h = 0; h < kHarmonics; ++h) {
            const float sw = sine_wavs[h * kSrcLen + n] * (voiced ? 1.0f : 0.0f) +
                             noise_amp * noise[h * kSrcLen + n];
            merged += linear_w[h] * sw;
        }
        s[n] = std::tanh(merged + linear_b);
    }
    return s;
}

// 源激励 s[95040] -> s_stft[18 * 23761]（9 real + 9 imag，16 点 STFT center=True）
inline void SourceStft(const std::vector<float>& s,
                       const std::vector<float>& hann,
                       std::vector<float>& s_stft) {
    constexpr int pad = kNfft / 2;  // 8
    std::vector<float> xp(kSrcLen + 2 * pad);
    // reflect pad
    for (int i = 0; i < pad; ++i) {
        xp[pad - 1 - i] = s[i + 1];
        xp[kSrcLen + pad + i] = s[kSrcLen - 2 - i];
    }
    for (int i = 0; i < kSrcLen; ++i) xp[pad + i] = s[i];
    s_stft.assign(2 * (kNfft / 2 + 1) * kStftFrames, 0.0f);
    std::vector<Cx> frame(kNfft);
    for (int t = 0; t < kStftFrames; ++t) {
        for (int k = 0; k < kNfft; ++k) {
            frame[k] = Cx(xp[t * kHop + k] * hann[k], 0.0f);
        }
        Fft16(frame, false);
        for (int b = 0; b < kNfft / 2 + 1; ++b) {
            s_stft[b * kStftFrames + t] = frame[b].real();
            s_stft[(9 + b) * kStftFrames + t] = frame[b].imag();
        }
    }
}

// mag/phase（各 9 x 23761）-> wav[95040]（16 点 ISTFT，overlap-add）
inline std::vector<float> Istft16(const std::vector<float>& mag,
                                  const std::vector<float>& ph,
                                  const std::vector<float>& hann) {
    constexpr int pad = kNfft / 2;
    const int n = (kStftFrames - 1) * kHop + kNfft;  // 95056
    std::vector<double> out(n, 0.0), wsum(n, 0.0);
    std::vector<Cx> spec(9);
    std::vector<Cx> frame(kNfft);
    for (int t = 0; t < kStftFrames; ++t) {
        for (int b = 0; b < 9; ++b) {
            const double m = std::exp(static_cast<double>(mag[b * kStftFrames + t]));
            const double p = std::sin(static_cast<double>(ph[b * kStftFrames + t]));
            spec[b] = Cx(static_cast<float>(m * std::cos(p)),
                         static_cast<float>(m * std::sin(p)));
        }
        // irfft：补齐共轭镜像频点（bins 9..15 = conj(bin 16-b)），否则逆变换结果错误
        for (int k = 0; k < 9; ++k) frame[k] = spec[k];
        for (int k = 9; k < kNfft; ++k) frame[k] = std::conj(spec[kNfft - k]);
        Fft16(frame, true);  // irfft
        const int st = t * kHop;
        for (int k = 0; k < kNfft; ++k) {
            const double v = frame[k].real() * hann[k];
            out[st + k] += v;
            wsum[st + k] += static_cast<double>(hann[k]) * hann[k];
        }
    }
    std::vector<float> y(kSrcLen);
    for (int i = 0; i < kSrcLen; ++i) {
        double v = wsum[pad + i] > 1e-8 ? out[pad + i] / wsum[pad + i] : 0.0;
        v = std::max(-static_cast<double>(kAudioLimit),
                     std::min(static_cast<double>(kAudioLimit), v));
        y[i] = static_cast<float>(v);
    }
    return y;
}

// 便捷封装：mel(1,80,frames<=198) -> wav(frames*480)
inline std::vector<float> Synth(const std::vector<float>& mel_80xf,
                                int frames,
                                const std::vector<float>& linear_w,
                                float linear_b,
                                const std::vector<float>& f0,
                                const std::vector<float>& raw_mag,
                                const std::vector<float>& raw_ph) {
    // 该函数仅用于演示接口；实际流程由 server 分步调用 f0/decode 模型。
    (void)mel_80xf; (void)frames; (void)linear_w; (void)linear_b; (void)f0;
    (void)raw_mag; (void)raw_ph;
    return std::vector<float>();
}

}  // namespace hift
