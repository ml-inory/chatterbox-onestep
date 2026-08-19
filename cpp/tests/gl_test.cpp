// 本地 DSP 自测（x86 可跑，无需 AX 芯片）：
//   1) FftAny(1920) 与 numpy 参考对比（fft_signal.bin / fft_ref.bin）
//   2) 真实 mel（mel_log10.bin）跑 Griffin-Lim -> gl_cpp.wav，并输出时长/RMS/耗时
// 编译：g++ -O2 -std=c++14 -I include tests/gl_test.cpp -o gl_test
#include "dsp.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<char> ReadBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<char>{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

float MaxErr(const std::vector<dsp::Cx>& a, const std::vector<dsp::Cx>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = ".";
    if (argc > 1) dir = argv[1];
    const std::string d = dir + "/";

    // 1) FFT 正确性
    {
        std::vector<char> sig_raw = ReadBin(d + "fft_signal.bin");
        std::vector<char> ref_raw = ReadBin(d + "fft_ref.bin");
        std::vector<float> sig(sig_raw.size() / 4);
        std::memcpy(sig.data(), sig_raw.data(), sig_raw.size());
        std::vector<dsp::Cx> x(sig.size());
        for (size_t i = 0; i < sig.size(); ++i) x[i] = dsp::Cx(sig[i], 0.0f);
        std::vector<dsp::Cx> y = dsp::FftAny(x);
        std::vector<dsp::Cx> ref(ref_raw.size() / 8);
        std::memcpy(ref.data(), ref_raw.data(), ref_raw.size());
        std::printf("FFT max err vs numpy: %.3e (%zu bins)\n", MaxErr(y, ref), y.size());
        // 往返：ifft(fft(x)) ~= x
        std::vector<dsp::Cx> z = dsp::IfftAny(y);
        float rt = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) rt = std::max(rt, std::abs(z[i] - x[i]));
        std::printf("FFT round-trip max err: %.3e\n", rt);
    }

    // 2) Griffin-Lim on real mel
    {
        std::vector<char> mel_raw = ReadBin(d + "mel_log10.bin");
        std::vector<char> inv_raw = ReadBin(d + "mel_inv_24k.bin");
        const int frames = static_cast<int>(mel_raw.size() / 4 / dsp::kNMels);
        std::vector<float> mel(mel_raw.size() / 4), mel_inv(inv_raw.size() / 4);
        std::memcpy(mel.data(), mel_raw.data(), mel_raw.size());
        std::memcpy(mel_inv.data(), inv_raw.data(), inv_raw.size());
        std::vector<float> win = dsp::Hann();

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> x = dsp::GriffinLim(mel, frames, mel_inv, win);
        auto t1 = std::chrono::steady_clock::now();
        float sec = std::chrono::duration<float>(t1 - t0).count();

        float rms = 0.0f, peak = 0.0f;
        for (float v : x) {
            rms += v * v;
            peak = std::max(peak, std::fabs(v));
        }
        rms = std::sqrt(rms / x.size());
        const float dur = static_cast<float>(x.size()) / dsp::kSr;
        std::printf("GL: frames=%d samples=%zu dur=%.2fs rms=%.4f peak=%.3f time=%.2fs\n",
                    frames, x.size(), dur, rms, peak, sec);

        std::vector<char> wav;
        dsp::WriteWav(x, &wav);
        std::ofstream f(d + "gl_cpp.wav", std::ios::binary);
        f.write(wav.data(), static_cast<std::streamsize>(wav.size()));
        std::printf("wrote %s/gl_cpp.wav (%zu bytes)\n", dir.c_str(), wav.size());
    }
    return 0;
}
