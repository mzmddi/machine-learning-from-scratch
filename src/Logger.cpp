// --- NOTES ---

// --- INCLUDES ---

#include <string>
#include <filesystem>
#include <chrono>

#include "Logger.h"

// --- CODE ---

Logger logger;

Logger::Logger()
{

    std::filesystem::create_directories("../logs");

    this->outfile.open("../logs/log.txt", std::ios::app);

    if (!this->outfile.is_open())
    {
        std::cout << "Could not open the log file" << std::endl;
        std::exit(0);
    }
}

void Logger::write(std::string m)
{
    // write in the file

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *local_time = std::localtime(&now_time);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

    std::string output_string = std::string(buffer) + " | " + m;

    std::cout << output_string << std::endl;

    if (this->outfile.is_open())
    {
        this->outfile << output_string << "\n";
        this->outfile.flush();
    }
}