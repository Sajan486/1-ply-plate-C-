#ifndef SETTINGS_H_
#define SETTINGS_H_

#include "Config.h"

class Settings {
public:
	static void apply(std::string fileSettingsPath, Config& config);

private:
	static bool parseTag(std::string input, std::string& output, std::string& val);
	static bool parseBool(std::string input);
	static std::vector<int> parseVector(std::string input);
};
#endif
