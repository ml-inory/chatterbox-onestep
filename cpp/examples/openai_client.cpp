// OpenAI-Compatible 板端 TTS 客户端（C++）。
// 用法：./openai_client --url http://127.0.0.1:8000/v1/audio/speech --tokens "12,34,56" --out out.wav

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string url = "http://127.0.0.1:8000/v1/audio/speech";
    std::string tokens_arg, tokens_file, out = "output.wav", fmt = "wav";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--url") && i + 1 < argc) url = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_arg = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens-file") && i + 1 < argc) tokens_file = argv[++i];
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--fmt") && i + 1 < argc) fmt = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--url U] --tokens \"1,2,3\"|--tokens-file f --out o [--fmt wav|mel]\n", argv[0]);
            return 1;
        }
    }

    std::vector<int> tokens;
    if (!tokens_arg.empty()) {
        std::stringstream ss(tokens_arg);
        std::string t;
        while (std::getline(ss, t, ',')) {
            if (!t.empty()) tokens.push_back(std::stoi(t));
        }
    } else if (!tokens_file.empty()) {
        std::ifstream f(tokens_file, std::ios::binary);
        std::vector<char> raw{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
        const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
        for (size_t i = 0; i < raw.size() / 4; ++i) {
            if (p[i] > 0) tokens.push_back(p[i]);
        }
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "no tokens\n");
        return 1;
    }

    std::string body = "{\"model\":\"chatterbox-onestep\",\"input\":[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        body += std::to_string(tokens[i]);
        if (i + 1 < tokens.size()) body += ",";
    }
    body += "],\"voice\":\"default\",\"response_format\":\"" + fmt + "\"}";

    // 解析 host:port/path
    std::string host_port = url.substr(url.find("://") + 3);
    std::string path = "/";
    size_t slash = host_port.find('/');
    if (slash != std::string::npos) {
        path = host_port.substr(slash);
        host_port = host_port.substr(0, slash);
    }
    size_t colon = host_port.rfind(':');
    std::string host = host_port.substr(0, colon);
    int port = colon == std::string::npos ? 80 : std::atoi(host_port.substr(colon + 1).c_str());

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1 || connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("connect");
        return 1;
    }
    std::string req = "POST " + path + " HTTP/1.1\r\nHost: " + host_port +
                      "\r\nContent-Type: application/json\r\nContent-Length: " +
                      std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send(fd, req.data(), req.size(), 0);

    std::vector<char> resp;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf), 0)) > 0) {
        resp.insert(resp.end(), buf, buf + n);
    }
    close(fd);

    size_t sep = std::string(resp.data(), resp.size()).find("\r\n\r\n");
    if (sep == std::string::npos) return 1;
    std::vector<char> audio(resp.begin() + static_cast<long>(sep) + 4, resp.end());
    std::ofstream f(out, std::ios::binary);
    f.write(audio.data(), static_cast<std::streamsize>(audio.size()));
    std::printf("OK: %zu bytes -> %s\n", audio.size(), out.c_str());
    return 0;
}
