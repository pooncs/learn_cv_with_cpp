#pragma once

#ifdef USE_SPDLOG
    #include "spdlog/spdlog.h"
    #define LOG_INFO(...) spdlog::info(__VA_ARGS__)
    #define LOG_WARN(...) spdlog::warn(__VA_ARGS__)
    
    inline void init_logging(const char* name) {
        // Spdlog auto-init
    }

#elif defined(USE_GLOG)
    #include <glog/logging.h>
    #include <fmt/format.h>
    
    // GLog doesn't support format strings natively, so we use fmt
    #define LOG_INFO(...) LOG(INFO) << fmt::format(__VA_ARGS__)
    #define LOG_WARN(...) LOG(WARNING) << fmt::format(__VA_ARGS__)

    inline void init_logging(const char* name) {
        google::InitGoogleLogging(name);
        FLAGS_logtostderr = 1;
    }
#else
    #include <iostream>
    #define LOG_INFO(...) std::cout << "[INFO] " << __VA_ARGS__ << "\n"
    #define LOG_WARN(...) std::cout << "[WARN] " << __VA_ARGS__ << "\n"
    inline void init_logging(const char* name) {}
#endif
