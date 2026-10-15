//
//  qwen3_tts_demo.cpp
//
//  Qwen3-TTS command line demo built on the LLM engine Omni/Talker path.
//

#include "llm/llm.hpp"
#include "audio/audio.hpp"
#include "httplib.h"
#include "jsonhpp/json.hpp"

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace MNN::Transformer;
using namespace MNN::Express;

constexpr int kSampleRate = 24000;
constexpr int kDefaultMaxFrames = 128;
constexpr int kMaxServerFrames = 2048;
constexpr size_t kMaxRequestBytes = 16384;
constexpr size_t kMaxInputBytes = 4096;
constexpr size_t kHttpWorkers = 4;
constexpr size_t kMaxQueuedRequests = 16;

static std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir[dir.size() - 1] == '/' || dir[dir.size() - 1] == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

static bool ensureDirectory(const std::string& path) {
    if (path.empty()) {
        return true;
    }
#ifdef _WIN32
    if (_mkdir(path.c_str()) == 0 || errno == EEXIST) {
        return true;
    }
#else
    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
#endif
    MNN_ERROR("failed to create dump_dir: %s\n", path.c_str());
    return false;
}

static bool dumpFloatVector(const std::string& path, const std::vector<float>& values) {
    std::ofstream os(path.c_str(), std::ios::binary);
    os.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    return os.good();
}

static std::vector<float> normalizeWaveform(const std::vector<float>& waveform, float targetPeak) {
    if (targetPeak <= 0.0f) {
        return waveform;
    }
    float peak = 0.0f;
    for (float value : waveform) {
        peak = std::max(peak, std::fabs(value));
    }
    if (peak <= 0.0f) {
        MNN_PRINT("normalize target_peak=%.6g original_peak=0 scale=1\n", targetPeak);
        return waveform;
    }
    const float scale = targetPeak / peak;
    std::vector<float> result(waveform.size());
    for (int i = 0; i < waveform.size(); ++i) {
        result[i] = std::max(-1.0f, std::min(1.0f, waveform[i] * scale));
    }
    MNN_PRINT("normalize target_peak=%.6g original_peak=%.6g scale=%.6g\n", targetPeak, peak, scale);
    return result;
}

static void printWaveformSummary(const char* name, const std::vector<float>& waveform, int sampleCount) {
    MNN_PRINT("%s shape=[1,%zu] size=%zu\n", name, waveform.size(), waveform.size());
    int n = std::min<int>(sampleCount, waveform.size());
    MNN_PRINT("%s first%d:", name, n);
    for (int i = 0; i < n; ++i) {
        MNN_PRINT(" %.8g", waveform[i]);
    }
    MNN_PRINT("\n");
}

static bool isOption(const char* arg) {
    return std::strncmp(arg, "--", 2) == 0;
}

static int runTextMode(const std::string& modelDir, const std::string& text, const std::string& language, int maxFrames,
                       const std::string& dumpDir, float normalizePeak, const std::string& refAudio,
                       const std::string& outputPath) {
    std::unique_ptr<Llm> llm(Llm::createLLM(joinPath(modelDir, "config.json")));
    llm->set_config("{\"tmp_path\":\"tmp\",\"async\":false}");
    if (!llm->load()) {
        MNN_ERROR("Qwen3-TTS load failed\n");
        return 1;
    }

    std::vector<float> waveform;
    MNN::Timer generationTimer;
    double firstAudioS = 0.0;
    size_t audioChunks = 0;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool lastChunk) {
        if (ptr && size > 0) {
            if (audioChunks == 0) {
                firstAudioS = generationTimer.durationInUs() / 1e6;
            }
            ++audioChunks;
            waveform.insert(waveform.end(), ptr, ptr + size);
        }
        return true;
    });

    generationTimer.reset();
    if (!llm->generateTTS(text, language, maxFrames, refAudio)) {
        MNN_ERROR("Qwen3-TTS generation failed\n");
        return 1;
    }
    const double generationS = generationTimer.durationInUs() / 1e6;
    if (waveform.empty()) {
        MNN_ERROR("Qwen3-TTS generated empty waveform\n");
        return 1;
    }

    printWaveformSummary("waveform", waveform, 16);
    std::vector<float> waveformToSave = waveform;
    if (normalizePeak > 0.0f) {
        waveformToSave = normalizeWaveform(waveform, normalizePeak);
        printWaveformSummary("waveform_normalized", waveformToSave, 16);
    }

    const auto* context = llm->getContext();
    if (!dumpDir.empty()) {
        if (!ensureDirectory(dumpDir)) {
            return 1;
        }
        dumpFloatVector(joinPath(dumpDir, "mnn_text_waveform.bin"), waveform);
        if (context && !context->output_tokens.empty()) {
            std::ofstream codesOs(joinPath(dumpDir, "mnn_text_codes.bin").c_str(), std::ios::binary);
            codesOs.write(reinterpret_cast<const char*>(context->output_tokens.data()),
                          context->output_tokens.size() * sizeof(int));
        }
        if (normalizePeak > 0.0f) {
            dumpFloatVector(joinPath(dumpDir, "mnn_text_waveform_normalized.bin"), waveformToSave);
        }
    }
    std::string wavPath = outputPath;
    if (wavPath.empty() && !dumpDir.empty()) {
        wavPath = joinPath(dumpDir, "qwen3_tts_text.wav");
    }
    if (!wavPath.empty()) {
        auto waveformVar =
            _Const(waveformToSave.data(), {static_cast<int>(waveformToSave.size())}, NCHW, halide_type_of<float>());
        if (!MNN::AUDIO::save(wavPath, waveformVar, kSampleRate)) {
            MNN_ERROR("failed to save wav: %s\n", wavPath.c_str());
            return 1;
        }
        MNN_PRINT("saved wav: %s\n", wavPath.c_str());
    }
    MNN_PRINT("Qwen3-TTS text C++ chain finished. frames=%d\n", context ? context->gen_seq_len : 0);
    MNN_PRINT("time to first audio (TTFA) = %.3f s\n generation wall time = %.3f s\n audio chunks = %zu\n",
              firstAudioS, generationS, audioChunks);
    if (context) {
        const double prefillS = context->prefill_us / 1e6;
        const double decodeS   = context->decode_us / 1e6;
        const double audioS    = context->audio_us / 1e6;
        const double wavDur    = static_cast<double>(waveform.size()) / kSampleRate;
        const double totalS    = prefillS + decodeS + audioS;
        MNN_PRINT("prefill time = %.2f s\n decode time = %.2f s\n audio process time = %.2f s\n", prefillS, decodeS,
                  audioS);
        MNN_PRINT("waveform duration = %.2f s\n audio RTF = %.3f\n", wavDur, wavDur > 0.0 ? totalS / wavDur : 0.0);
    }
    return 0;
}

static void printServerUsage(const char* program) {
    MNN_PRINT("Usage: %s <model_dir> --serve --ref_audio <wav> [--host 127.0.0.1] [--port 8000] "
              "[--model qwen3-tts] [--voice default] [--max_frames %d]\n", program, kDefaultMaxFrames);
    MNN_PRINT("Serves POST /v1/audio/speech (WAV only), GET /v1/models, and GET /health.\n");
    MNN_PRINT("No authentication is provided; use a trusted interface or an authenticated reverse proxy.\n");
}

static bool parsePositiveInt(const char* text, int maximum, int& result) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > maximum) {
        return false;
    }
    result = static_cast<int>(value);
    return true;
}

static void setApiError(httplib::Response& response, int status, const std::string& message,
                        const std::string& parameter = "") {
    using nlohmann::json;
    json error = {{"error", {{"message", message},
                             {"type", status >= 500 ? "server_error" : "invalid_request_error"},
                             {"param", parameter.empty() ? json(nullptr) : json(parameter)},
                             {"code", nullptr}}}};
    response.status = status;
    response.set_content(error.dump(), "application/json");
}

struct ServerOptions {
    std::string host = "127.0.0.1";
    std::string model = "qwen3-tts";
    std::string voice = "default";
    std::string refAudio;
    int port = 8000;
    int maxFrames = kDefaultMaxFrames;
    bool help = false;
};

static bool parseServerOptions(int argc, char** argv, ServerOptions& options) {
    for (int index = 3; index < argc; index += 2) {
        const std::string option = argv[index];
        if (option == "--help") {
            options.help = true;
            return true;
        }
        if (index + 1 >= argc || isOption(argv[index + 1]) || argv[index + 1][0] == '\0') {
            MNN_ERROR("%s requires a value\n", option.c_str());
            return false;
        }
        const char* value = argv[index + 1];
        if (option == "--host") {
            options.host = value;
        } else if (option == "--model") {
            options.model = value;
        } else if (option == "--voice") {
            options.voice = value;
        } else if (option == "--ref_audio") {
            options.refAudio = value;
        } else if (option == "--port") {
            if (!parsePositiveInt(value, 65535, options.port)) {
                MNN_ERROR("--port must be an integer in [1, 65535]\n");
                return false;
            }
        } else if (option == "--max_frames") {
            if (!parsePositiveInt(value, kMaxServerFrames, options.maxFrames)) {
                MNN_ERROR("--max_frames must be an integer in [1, %d]\n", kMaxServerFrames);
                return false;
            }
        } else {
            MNN_ERROR("unknown server option: %s\n", option.c_str());
            return false;
        }
    }
    if (options.refAudio.empty()) {
        MNN_ERROR("--ref_audio is required to configure the server voice\n");
        printServerUsage(argv[0]);
        return false;
    }
    return true;
}

static bool parseSpeechRequest(const std::string& requestBody, const ServerOptions& options, std::string& text,
                               httplib::Response& response) {
    const auto body = nlohmann::json::parse(requestBody, nullptr, false);
    if (!body.is_object()) {
        setApiError(response, 400, "Request body must be a valid JSON object.");
        return false;
    }
    for (auto item = body.begin(); item != body.end(); ++item) {
        const auto& field = item.key();
        if (field != "model" && field != "input" && field != "voice" && field != "response_format" &&
            field != "speed" && field != "stream") {
            setApiError(response, 400, "Unsupported request field.", field);
            return false;
        }
    }
    for (const char* field : {"model", "input", "voice"}) {
        auto value = body.find(field);
        if (value == body.end() || !value->is_string() || value->get_ref<const std::string&>().empty()) {
            setApiError(response, 400, "A non-empty string is required.", field);
            return false;
        }
    }
    const auto& input = body["input"].get_ref<const std::string&>();
    if (input.size() > kMaxInputBytes ||
        std::all_of(input.begin(), input.end(), [](unsigned char value) { return std::isspace(value); })) {
        setApiError(response, 400,
                    "Input must contain text and be at most " + std::to_string(kMaxInputBytes) + " UTF-8 bytes.",
                    "input");
        return false;
    }
    if (body["model"].get_ref<const std::string&>() != options.model) {
        setApiError(response, 404, "The requested model is not loaded.", "model");
        return false;
    }
    if (body["voice"].get_ref<const std::string&>() != options.voice) {
        setApiError(response, 400, "The requested voice is not configured.", "voice");
        return false;
    }
    if (body.contains("response_format") &&
        (!body["response_format"].is_string() || body["response_format"].get<std::string>() != "wav")) {
        setApiError(response, 400, "Only WAV output is supported.", "response_format");
        return false;
    }
    if (body.contains("speed") && (!body["speed"].is_number() || body["speed"].get<double>() != 1.0)) {
        setApiError(response, 400, "Only speed=1.0 is supported.", "speed");
        return false;
    }
    if (body.contains("stream") && (!body["stream"].is_boolean() || body["stream"].get<bool>())) {
        setApiError(response, 400, "Streaming is not supported.", "stream");
        return false;
    }
    text = input;
    return true;
}

static int runServerMode(int argc, char** argv) {
    using nlohmann::json;
    ServerOptions options;
    if (!parseServerOptions(argc, argv, options)) {
        return 1;
    }
    if (options.help) {
        printServerUsage(argv[0]);
        return 0;
    }

    std::unique_ptr<Llm> llm(Llm::createLLM(joinPath(argv[1], "config.json")));
    const auto modelConfig = json::parse(llm->dump_config(), nullptr, false);
    if (!modelConfig.is_object() || modelConfig.value("has_talker", json(false)) != true ||
        (modelConfig.value("talker_type", json("")) != "qwen3_tts" &&
         modelConfig.value("model_type", json("")) != "qwen3_tts")) {
        MNN_ERROR("server requires a Qwen3-TTS model with has_talker=true\n");
        return 1;
    }
    {
        auto reference = MNN::AUDIO::load(options.refAudio, kSampleRate);
        if (!reference.first.get() || !reference.first->getInfo() || reference.first->getInfo()->size <= 0) {
            MNN_ERROR("failed to load reference audio: %s\n", options.refAudio.c_str());
            return 1;
        }
    }
    llm->set_config("{\"tmp_path\":\"tmp\",\"async\":false}");
    if (!llm->load()) {
        MNN_ERROR("Qwen3-TTS load failed\n");
        return 1;
    }

    std::mutex inferenceMutex;
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(kHttpWorkers, kMaxQueuedRequests); };
    server.set_payload_max_length(kMaxRequestBytes);
    server.set_read_timeout(10, 0);
    server.set_write_timeout(60, 0);
    server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (response.body.empty()) {
            setApiError(response, response.status,
                        response.status == 413 ? "Request body exceeds " + std::to_string(kMaxRequestBytes) + " bytes."
                                               : "HTTP request failed.");
        }
    });
    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{\"status\":\"ok\"}", "application/json");
    });
    server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& response) {
        json models = {{"object", "list"},
                       {"data", json::array({{{"id", options.model}, {"object", "model"}, {"created", 0},
                                               {"owned_by", "mnn"}}})}};
        response.set_content(models.dump(), "application/json");
    });
    server.Post("/v1/audio/speech", [&](const httplib::Request& request, httplib::Response& response) {
        std::string text;
        if (!parseSpeechRequest(request.body, options, text, response)) {
            return;
        }

        std::unique_lock<std::mutex> lock(inferenceMutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            response.set_header("Retry-After", "1");
            setApiError(response, 503, "Speech generation is busy; retry later.");
            return;
        }
        std::vector<float> waveform;
        MNN::Timer generationTimer;
        double firstAudioMs = 0.0;
        llm->setWavformCallback([&](const float* samples, size_t size, bool) {
            if (samples && size > 0) {
                if (waveform.empty()) {
                    firstAudioMs = generationTimer.durationInUs() / 1e3;
                }
                waveform.insert(waveform.end(), samples, samples + size);
            }
            return true;
        });
        generationTimer.reset();
        const bool generated = llm->generateTTS(text, "auto", options.maxFrames, options.refAudio);
        const double generationMs = generationTimer.durationInUs() / 1e3;
        llm->setWavformCallback(nullptr);
        if (!generated || waveform.empty() ||
            !std::all_of(waveform.begin(), waveform.end(), [](float value) { return std::isfinite(value); })) {
            setApiError(response, 500, "Speech generation failed.");
            return;
        }
        auto waveformVar = _Const(waveform.data(), {static_cast<int>(waveform.size())}, NCHW, halide_type_of<float>());
        std::ostringstream wav(std::ios::out | std::ios::binary);
        if (!MNN::AUDIO::save(wav, waveformVar, kSampleRate)) {
            setApiError(response, 500, "WAV encoding failed.");
            return;
        }
        response.set_header("X-TTFA-Ms", std::to_string(firstAudioMs));
        response.set_header("X-Generation-Ms", std::to_string(generationMs));
        response.set_header("Cache-Control", "no-store");
        response.set_content(wav.str(), "audio/wav");
    });

    MNN_PRINT("Qwen3-TTS server: http://%s:%d (model=%s, voice=%s, max_frames=%d)\n", options.host.c_str(), options.port,
              options.model.c_str(), options.voice.c_str(), options.maxFrames);
    MNN_PRINT("No authentication is enabled.\n");
    if (!server.listen(options.host.c_str(), options.port)) {
        MNN_ERROR("failed to listen on %s:%d\n", options.host.c_str(), options.port);
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[2], "--serve") == 0) {
        return runServerMode(argc, argv);
    }
    if (argc < 4 || std::strcmp(argv[2], "--text") != 0) {
        printServerUsage(argv[0]);
        MNN_PRINT(
            "Usage: %s <model_dir> --text <text> [max_frames] [dump_dir] [language] "
            "--ref_audio <wav> [--output <wav>] [--normalize [target_peak]]\n",
            argv[0]);
        MNN_PRINT(
            "--ref_audio is required for speaker-embedding-only voice cloning; the zero speaker embedding path is "
            "unsupported.\n");
        MNN_PRINT("--output overrides the dump_dir WAV path; its parent directory must already exist.\n");
        MNN_PRINT("max_frames defaults to %d and is a hard cap; too small a value truncates the utterance.\n",
              kDefaultMaxFrames);
        MNN_PRINT("Download ref audio to: transformers/llm/resource/audio/qwen3_tts_ref.wav\n");
        MNN_PRINT(
            "Download URL: https://modelscope.cn/datasets/huangzhengxiang/qwen3-tts-ref/resolve/master/"
            "qwen3_tts_ref.wav\n");
        return 1;
    }
    std::string modelDir = argv[1];
    std::string text = argv[3];
    int maxFrames = kDefaultMaxFrames;
    std::string dumpDir;
    std::string language = "auto";
    float normalizePeak = -1.0f;
    bool normalizeRequested = false;
    std::string refAudio;
    std::string outputPath;

    int index = 4;
    if (index < argc && !isOption(argv[index])) {
        maxFrames = std::atoi(argv[index++]);
    }
    if (index < argc && !isOption(argv[index])) {
        dumpDir = argv[index++];
    }
    if (index < argc && !isOption(argv[index])) {
        language = argv[index++];
    }
    while (index < argc) {
        if (std::strcmp(argv[index], "--output") == 0) {
            ++index;
            if (index >= argc || isOption(argv[index]) || argv[index][0] == '\0') {
                MNN_ERROR("--output requires a wav path\n");
                return 1;
            }
            outputPath = argv[index++];
            continue;
        }
        if (std::strcmp(argv[index], "--ref_audio") == 0) {
            ++index;
            if (index >= argc || isOption(argv[index])) {
                MNN_ERROR("--ref_audio requires a wav path\n");
                return 1;
            }
            refAudio = argv[index++];
            continue;
        }
        if (std::strcmp(argv[index], "--normalize") == 0) {
            normalizeRequested = true;
            normalizePeak = 1.0f;
            ++index;
            if (index < argc && !isOption(argv[index])) {
                normalizePeak = static_cast<float>(std::atof(argv[index++]));
            }
            continue;
        }
        MNN_ERROR("unknown option: %s\n", argv[index]);
        return 1;
    }
    if (maxFrames <= 0) {
        MNN_ERROR("max_frames must be positive\n");
        return 1;
    }
    if (refAudio.empty()) {
        MNN_ERROR("--ref_audio is required for Qwen3-TTS speaker-embedding-only voice cloning\n");
        return 1;
    }
    if (normalizeRequested && (normalizePeak <= 0.0f || normalizePeak > 1.0f)) {
        MNN_ERROR("normalize target_peak must be in (0, 1]\n");
        return 1;
    }
    return runTextMode(modelDir, text, language, maxFrames, dumpDir, normalizePeak, refAudio, outputPath);
}
