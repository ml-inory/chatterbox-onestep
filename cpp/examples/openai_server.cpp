// OpenAI-Compatible 板端 TTS 服务（C++，torch-free / ffmpeg-free / 无第三方 DSP 依赖）。
//
// 仅依赖 AX Engine（ax_engine/ax_sys）+ 标准库：
//   input(S3 speech tokens) --S3Gen AXMODEL--> mel --dsp::GriffinLim(自实现 Bluestein FFT)--> wav
//
// 用法：
//   ./openai_server --model models/model.axmodel --assets . --port 8000
//   ./openai_client --url http://127.0.0.1:8000/v1/audio/speech --tokens "12,34,56" --out out.wav
//
// --assets 目录需包含：mel_inv_24k.bin（961x80 float32，线性谱逆变换矩阵）
//                      default_embedding.bin（192 float32，内置音色 xvector）

#include "dsp.hpp"
#include "model_runner.hpp"
#include "hift_dsp.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------- 极简 JSON 提取（固定 schema） ----------
std::vector<int> ParseTokenList(const std::string& body) {
    std::vector<int> out;
    size_t p = body.find("\"input\"");
    if (p == std::string::npos) return out;
    p = body.find('[', p);
    if (p == std::string::npos) return out;
    size_t q = body.find(']', p);
    std::string arr = body.substr(p + 1, q - p - 1);
    std::stringstream ss(arr);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            out.push_back(std::stoi(tok));
        } catch (...) {
        }
    }
    return out;
}

std::string ParseStringField(const std::string& body, const std::string& key) {
    size_t p = body.find("\"" + key + "\"");
    if (p == std::string::npos) return "";
    p = body.find(':', p);
    p = body.find('"', p);
    if (p == std::string::npos) return "";
    ++p;
    size_t q = body.find('"', p);
    if (q == std::string::npos) return "";
    return body.substr(p, q - p);
}

std::vector<float> ParseEmbedding(const std::string& body) {
    std::vector<float> out;
    size_t p = body.find("\"embedding\"");
    if (p == std::string::npos) return out;
    p = body.find('[', p);
    if (p == std::string::npos) return out;
    size_t q = body.find(']', p);
    std::string arr = body.substr(p + 1, q - p - 1);
    std::stringstream ss(arr);
    std::string tok;
    while (std::getline(ss, tok, ',') && out.size() < 192) {
        try {
            out.push_back(std::stof(tok));
        } catch (...) {
        }
    }
    return out;
}

// ---------- 极简 HTTP（单连接顺序处理） ----------
struct HttpResponse {
    int code = 200;
    std::string ctype = "application/json";
    std::vector<char> body;
};

void SendResponse(int fd, const HttpResponse& resp) {
    std::string head =
        "HTTP/1.1 " + std::to_string(resp.code) + " " +
        (resp.code == 200 ? "OK" : resp.code == 400 ? "Bad Request"
                                : resp.code == 404 ? "Not Found"
                                                   : "Internal Server Error") +
        "\r\nContent-Type: " + resp.ctype +
        "\r\nContent-Length: " + std::to_string(resp.body.size()) +
        "\r\nConnection: close\r\n\r\n";
    send(fd, head.data(), head.size(), 0);
    if (!resp.body.empty()) send(fd, resp.body.data(), resp.body.size(), 0);
}

HttpResponse HandleRequest(ModelRunner& runner, ModelRunner* f0_runner, ModelRunner* dec_runner,
                           const std::vector<float>& linear_w, float linear_b,
                           const std::vector<float>& mel_inv,
                           const std::vector<float>& default_emb,
                           const std::vector<float>& win,
                           const std::vector<float>& hann16,
                           const std::string& method, const std::string& path,
                           const std::string& body) {
    using namespace dsp;
    if (method == "GET" && path == "/v1/models") {
        const char* j = "{\"object\":\"list\",\"data\":[{\"id\":\"chatterbox-onestep\","
                        "\"object\":\"model\",\"created\":0,\"owned_by\":\"axera\"}]}";
        HttpResponse r;
        r.body.assign(j, j + std::strlen(j));
        return r;
    }
    if (method != "POST" || path != "/v1/audio/speech") {
        HttpResponse r;
        r.code = 404;
        r.body = std::vector<char>{'n', 'o', 't', ' ', 'f', 'o', 'u', 'n', 'd'};
        return r;
    }
    std::vector<int> tokens = ParseTokenList(body);
    const std::string fmt = ParseStringField(body, "response_format");
    if (tokens.empty() || (fmt != "wav" && fmt != "" && fmt != "mel")) {
        HttpResponse r;
        r.code = 400;
        r.body = std::vector<char>{'b', 'a', 'd', ' ', 'r', 'e', 'q', 'u', 'e', 's', 't'};
        return r;
    }
    const std::string fmt_out = fmt.empty() ? "wav" : fmt;

    // 输入组装：tokens(int32, pad 256) / token_len(int32) / embedding(float32) / z(float32)
    std::vector<char> tokens_raw(256 * 4, 0);
    const int n = std::min(static_cast<int>(tokens.size()), 256);
    for (int i = 0; i < n; ++i) {
        int32_t v = std::max(0, std::min(6560, tokens[i]));
        std::memcpy(tokens_raw.data() + 4 * i, &v, 4);
    }
    std::vector<char> len_raw(4, 0);
    int32_t len_v = n;
    std::memcpy(len_raw.data(), &len_v, 4);

    std::vector<float> emb = ParseEmbedding(body);
    if (emb.size() != 192) emb = default_emb;
    std::vector<char> emb_raw(192 * 4);
    std::memcpy(emb_raw.data(), emb.data(), 192 * 4);

    std::vector<char> z_raw(80 * 512 * 4);
    std::mt19937 gen(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* zf = reinterpret_cast<float*>(z_raw.data());
    for (size_t i = 0; i < 80 * 512; ++i) zf[i] = nd(gen);

    std::vector<std::vector<char>> inputs{tokens_raw, len_raw, emb_raw, z_raw};
    std::vector<std::vector<float>> outputs = runner.RunBytes(inputs);
    const std::vector<float>& mel = outputs.at(0);  // 1x80x512
    const int frames = n * 2;
    std::vector<float> mel_log10(kNMels * frames);
    for (int m = 0; m < kNMels; ++m) {
        for (int t = 0; t < frames; ++t) mel_log10[m * frames + t] = mel[m * 512 + t];
    }

    HttpResponse r;
    if (fmt_out == "mel") {
        std::string j = "{\"mel\":[";
        for (int t = 0; t < frames; ++t) {
            j += "[";
            for (int m = 0; m < kNMels; ++m) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.6g", mel_log10[m * frames + t]);
                j += buf;
                if (m + 1 < kNMels) j += ",";
            }
            j += "]";
            if (t + 1 < frames) j += ",";
        }
        j += "]}";
        r.body.assign(j.begin(), j.end());
        return r;
    }
    std::vector<float> x;
    if (f0_runner != nullptr && dec_runner != nullptr) {
        // ---- HiFT 神经声码器（f0/decode 两个 AXMODEL + 宿主 DSP）----
        constexpr int kTMel = hift::kTMel;
        std::vector<float> mel_pad(80 * kTMel, hift::kPadVal);
        for (int m = 0; m < dsp::kNMels; ++m) {
            for (int t = 0; t < frames; ++t) mel_pad[m * kTMel + t] = mel_log10[m * frames + t];
        }
        std::vector<std::vector<float>> f0_out = f0_runner->Run({mel_pad});
        const std::vector<float>& f0 = f0_out.at(0);  // 198

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> ud(-static_cast<float>(M_PI), static_cast<float>(M_PI));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> phase(hift::kHarmonics, 0.0f);
        std::vector<float> noise(hift::kHarmonics * hift::kSrcLen);
        for (int h = 0; h < hift::kHarmonics; ++h) phase[h] = ud(gen);
        phase[0] = 0.0f;
        for (float& v : noise) v = nd(gen);

        std::vector<float> s = hift::SourceFromF0(f0, phase, noise, linear_w, linear_b);
        std::vector<float> s_stft;
        hift::SourceStft(s, hann16, s_stft);
        std::vector<std::vector<float>> dec_out = dec_runner->Run({mel_pad, s_stft});
        const std::vector<float>& raw_mag = dec_out.at(0);  // 9 x 23761
        const std::vector<float>& raw_ph = dec_out.at(1);
        x = hift::Istft16(raw_mag, raw_ph, hann16);
        x.resize(static_cast<size_t>(frames) * hift::kUpsample);
    } else {
        x = GriffinLim(mel_log10, frames, mel_inv, win);
    }
    WriteWav(x, &r.body);
    r.ctype = "audio/wav";
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "models/model.axmodel";
    std::string f0_model = "models/hifift_f0.axmodel";
    std::string dec_model = "models/hifift_decode.axmodel";
    std::string assets_dir = ".";
    int port = 8000;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--f0-model") && i + 1 < argc) f0_model = argv[++i];
        else if (!std::strcmp(argv[i], "--decode-model") && i + 1 < argc) dec_model = argv[++i];
        else if (!std::strcmp(argv[i], "--assets") && i + 1 < argc) assets_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--model m.axmodel] [--f0-model f.axmodel] [--decode-model d.axmodel] [--assets dir] [--port N]\n", argv[0]);
            return 1;
        }
    }

    auto read_bin = [&](const std::string& name) {
        std::ifstream f(assets_dir + "/" + name, std::ios::binary);
        if (!f) throw std::runtime_error("missing asset: " + name);
        return std::vector<char>(
            std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    };
    const std::vector<char> inv_raw = read_bin("mel_inv_24k.bin");
    const std::vector<char> emb_raw = read_bin("default_embedding.bin");
    const std::vector<char> lw_raw = read_bin("hift_linear_w.bin");
    const std::vector<char> lb_raw = read_bin("hift_linear_b.bin");
    if (inv_raw.size() != dsp::kNBins * dsp::kNMels * 4 || emb_raw.size() != 192 * 4) {
        throw std::runtime_error("asset size mismatch");
    }
    std::vector<float> mel_inv(dsp::kNBins * dsp::kNMels);
    std::vector<float> default_emb(192);
    std::vector<float> linear_w(hift::kHarmonics);
    float linear_b = 0.0f;
    std::memcpy(mel_inv.data(), inv_raw.data(), inv_raw.size());
    std::memcpy(default_emb.data(), emb_raw.data(), emb_raw.size());
    std::memcpy(linear_w.data(), lw_raw.data(), lw_raw.size());
    std::memcpy(&linear_b, lb_raw.data(), 4);

    ModelRunner runner(model_path, "model");
    ModelRunner* f0_runner = nullptr;
    ModelRunner* dec_runner = nullptr;
    try {
        f0_runner = new ModelRunner(f0_model, "hifift_f0");
        dec_runner = new ModelRunner(dec_model, "hifift_decode");
        std::fprintf(stderr, "[openai-server] vocoder: HiFT NPU (f0+decode)\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HiFT vocoder models unavailable, fallback to Griffin-Lim: %s\n", e.what());
        delete f0_runner; delete dec_runner; f0_runner = nullptr; dec_runner = nullptr;
    }
    const std::vector<float> win = dsp::Hann();
    const std::vector<float> hann16 = hift::Hann16();

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) return 1;
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        listen(listen_fd, 8) < 0) {
        std::perror("bind/listen");
        return 1;
    }
    std::printf("[openai-server] listening on port %d (C++, torch-free)\n", port);
    std::fflush(stdout);

    for (;;) {
        int fd = accept(listen_fd, nullptr, nullptr);
        if (fd < 0) continue;
        char req[8192];
        ssize_t got = recv(fd, req, sizeof(req) - 1, 0);
        if (got <= 0) {
            close(fd);
            continue;
        }
        req[got] = '\0';
        std::string request(req);
        std::string method, path, body;
        {
            std::istringstream ss(request);
            ss >> method >> path;
        }
        size_t hdr_end = request.find("\r\n\r\n");
        if (hdr_end != std::string::npos) {
            size_t cl = request.find("Content-Length:");
            size_t body_start = hdr_end + 4;
            if (cl != std::string::npos && cl < hdr_end) {
                size_t eol = request.find("\r\n", cl);
                int len = std::atoi(request.substr(cl + 15, eol - cl - 15).c_str());
                if (len > 0 && static_cast<int>(body_start + len) <= static_cast<int>(got)) {
                    body = request.substr(body_start, len);
                }
            }
        }
        HttpResponse resp;
        try {
            resp = HandleRequest(runner, f0_runner, dec_runner, linear_w, linear_b,
                                 mel_inv, default_emb, win, hann16, method, path, body);
        } catch (const std::exception& e) {
            resp.code = 500;
            resp.ctype = "text/plain";
            const std::string msg = std::string("synthesis failed: ") + e.what();
            resp.body.assign(msg.begin(), msg.end());
        }
        SendResponse(fd, resp);
        close(fd);
    }
    return 0;
}
