#include "Settings.h"

void Settings::apply(std::string fileSettingsPath, Config& config) {
	std::fstream fs;
	fs.open(fileSettingsPath, std::fstream::in | std::fstream::out);

	std::string line;
	std::vector<std::string> inputSettings;

	while(!fs.eof()) {
		std::getline(fs, line);
		inputSettings.push_back(line);
	}

	for (int param = 0; param < inputSettings.size(); param++) {
		std::string tag, val;
		std::string line = inputSettings[param];

		// Check to see if there could be a parameter present
		if (parseTag(line, tag, val)) {
			bool vectorParam = std::string::npos != val.find(":");

			if (!vectorParam) {
				// General params
				     if (tag == "validationMode")   	config.validationMode    = parseBool(val);
				else if (tag == "validationFolder")		config.validationFolder  = val;
				else if (tag == "doublePrecision")  	config.doublePrecision   = parseBool(val);
				else if (tag == "enableGPU")        	config.enableGPU         = parseBool(val);
				else if (tag == "cachePath")        	config.cachePath         = val;
				else if (tag == "plotOutputFolder") 	config.plotOutputFolder  = val;
				else if (tag == "cacheSize")        	config.cacheSize         = vat::MemorySize(std::stod(val), vat::GB);
				else if (tag == "maxIO")            	config.maxIO             = vat::MemorySize(std::stod(val), vat::GB);
				else if (tag == "maxBlockSize")     	config.maxBlockSize      = vat::MemorySize(std::stod(val), vat::GB);

			    // Geom params
				else if (tag == "NumSolidLay")          config.getGeom().NumSolidLay          = std::stod(val);
				else if (tag == "NumFluidLay")          config.getGeom().NumFluidLay          = std::stod(val);
				else if (tag == "NumSolidFluidIntrFc")  config.getGeom().NumSolidFluidIntrFc  = std::stod(val);
				else if (tag == "NumSourcePt_IntrFc_x") config.getGeom().NumSourcePt_IntrFc_x = std::stod(val);
				else if (tag == "NumSourcePt_IntrFc_y") config.getGeom().NumSourcePt_IntrFc_y = std::stod(val);
				else if (tag == "AngTestPt")            config.getGeom().AngTestPt            = std::stod(val);
				else if (tag == "Length_IntrFc_x")      config.getGeom().Length_IntrFc_x      = std::stod(val);
				else if (tag == "Dist_IntrFc1")         config.getGeom().Dist_IntrFc1         = std::stod(val);
				else if (tag == "Dist_IntrFc2")         config.getGeom().Dist_IntrFc2         = std::stod(val);
				//else if (tag == "ReceiverShift")       config::geom::ReceiverShift        = std::stod(val);

				     // Transducer params
				else if (tag == "NumTrans")          config.getTransducer().NumTrans          = std::stod(val);
				else if (tag == "InnerR_Trans")      config.getTransducer().InnerR_Trans      = std::stod(val);
				else if (tag == "OuterR_Trans")      config.getTransducer().OuterR_Trans      = std::stod(val);
				else if (tag == "NumSourcePt_Trans") config.getTransducer().NumSourcePt_Trans = std::stod(val);
				else if (tag == "IntrFcShift")       config.getTransducer().IntrFcShift       = std::stod(val);
				else if (tag == "Rotation_Trans1")   config.getTransducer().Rotation_Trans1   = std::stod(val);
				else if (tag == "Rotation_Trans2")   config.getTransducer().Rotation_Trans2   = std::stod(val);
				else if (tag == "freq")              config.getTransducer().freq              = std::stod(val);
				else if (tag == "Vso")               config.getTransducer().Vso               = std::stod(val);
				else if (tag == "Vto")               config.getTransducer().Vto               = std::stod(val);

				     // WaveField params
				else if (tag == "NumTarget_z")  config.getWavefield().NumTarget_z  = std::stod(val);
				else if (tag == "X_PlaneCoord") config.getWavefield().X_PlaneCoord = std::stod(val);
				else if (tag == "Y_PlaneCoord") config.getWavefield().Y_PlaneCoord = std::stod(val);
				else if (tag == "Z_PlaneCoord") config.getWavefield().Y_PlaneCoord = std::stod(val);

				     // Fluid params
				else if (tag == "Fluid_rho") config.getFluid().Fluid_rho = std::stod(val);
				else if (tag == "WaveVel_P") config.getFluid().WaveVel_P = std::stod(val);

				     // Solid params
				else if (tag == "Solid_rho") config.getSolid().Solid_rho = std::stod(val);

					// Time Domain params
				else if (tag == "SampRate")  config.getTimedomain().SampRate = std::stod(val);
				else if (tag == "NumSampPt") config.getTimedomain().NumSampPt = std::stod(val);
				else if (tag == "CentFreq")  config.getTimedomain().CentFreq = std::stod(val);
				else if (tag == "H") 		 config.getTimedomain().H = std::stod(val);
				else if (tag == "k") 		 config.getTimedomain().k = std::stod(val);
				else if (tag == "NumCycles") config.getTimedomain().NumCycles = std::stod(val);

			} else {
				//	 if (tag == "plotModes")     config.plotModes     = parseVector(val);
				 if (tag == "targetDevices") config.targetDevices = parseVector(val);
			}
		}
		else throw std::runtime_error("Invalid config file.");
	}
}

bool Settings::parseTag(std::string input, std::string& tag, std::string& val) {

	// remove comments
	if (std::string::npos != input.find("#")) {
		int commentIdx = input.find("#");
		input = input.substr(0, commentIdx);
	}

	bool validParam = false;
	if (std::string::npos != input.find(" ") && input.size() > 0) {

		int spaceIdx = input.find(" ");
		tag = input.substr(0, spaceIdx);
		val = input.substr(spaceIdx + 1, input.size() - spaceIdx - 1);
		while(val.substr(0, 1) == " ") {
			val = val.substr(1, val.size() - 1);
		}

		if (std::string::npos != val.find(" ")) {
			int lastSpaceIdx = val.find(" ");
			val = val.substr(0, lastSpaceIdx);
		}

		if (val.size() > 0) {
			validParam = true;
		}
	}
	return validParam;
}

bool Settings::parseBool(std::string input) {
	if (input == "0" || input == "false") return false;
	if (input == "1" || input == "true" ) return true;

	printf("Error: boolean parameter in settings had unexpected value\n");
	exit (EXIT_FAILURE);
}

std::vector<int> Settings::parseVector(std::string input) {
	input = input.substr(1, input.size() - 1);
	std::vector<int> output;
	while(input.size() > 0) {

		// see if this is the last element
		if (std::string::npos == input.find(",")) {

			// If there are leading spaces, truncate
			if (std::string::npos != input.find(" ")) {
				input = input.substr(0, input.find(" "));
			}
			if (input.size() > 0) {
				output.push_back(stoi(input));
				input = "";
			}
		} else {

			int commaIdx = input.find(",");
			output.push_back(stoi(input.substr(0, commaIdx)));
			input = input.substr(commaIdx + 1, input.size() - commaIdx - 1);
		}
	}
	return output;
}
