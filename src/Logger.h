#ifndef LOGGER_H
#define LOGGER_H

// --- NOTES ---

// --- INCLUDES ---
#include <iostream>
#include <string>
#include <fstream>

// --- CODE ---

class Logger
{
    std::ofstream outfile;

public:
    Logger();

    // --- LOGIC ---

    void write(std::string m);
};

extern Logger logger;

#endif