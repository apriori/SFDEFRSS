#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "./System/System.h"
#include <filesystem>
#include <vector>

using namespace std;


// Load Images from Dir
inline int getdir (const std::filesystem::path& dir, std::vector<std::filesystem::path> &files)
{
    const std::filesystem::path directory{dir};

    for (auto const& dir_entry : std::filesystem::directory_iterator{directory})
    {
        if(!dir_entry.is_regular_file()) {
            continue;
        }

        files.push_back(dir_entry.path());
    }

    std::sort(files.begin(), files.end());
    return files.size();
}


inline int getfiles (const std::filesystem::path& dir, std::vector<std::filesystem::path> &files)
{
    const std::filesystem::path directory{dir};

    for (auto const& dir_entry : std::filesystem::directory_iterator{directory})
    {
        if(!dir_entry.is_regular_file()) {
            continue;
        }

        files.push_back(dir_entry.path().stem());
    }

    std::sort(files.begin(), files.end());
    return files.size();
}

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./build/SFRSS target_path mode" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<std::filesystem::path> leftImagePaths;
    vector<std::filesystem::path> rightImagePaths;
    vector<std::filesystem::path> imageNames;
    vector<double> vTimestamps;
    std::filesystem::path leftDirPath = std::filesystem::path(argv[1]) / "frames" / "cam0";
    std::filesystem::path rightDirPath = std::filesystem::path(argv[1]) / "frames" / "cam1";
    int mode = atoi(argv[2]);

    getdir(leftDirPath, leftImagePaths);
    getdir(rightDirPath, rightImagePaths);
    getfiles(leftDirPath, imageNames);

    int nPairs = leftImagePaths.size();
    char buffer[256] = {0};
    for (unsigned int ii = 0; ii < leftImagePaths.size(); ++ii){
        memset(buffer, 0, 256);
        memcpy(buffer, imageNames[ii].c_str(), imageNames[ii].string().size() - 4);
        vTimestamps.push_back(atof(buffer));
    }

    SFRSS::System sys(string(argv[1]) + "/SFRSS.yaml", argv[1], 2);

    vector<float> vTimesTrack;
    vTimesTrack.resize(nPairs);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nPairs << endl << endl;
    cout << "mode: " << mode << endl;

    // Main loop
    cv::Mat leftImage, rightImage;
    for(int ni = 0; ni < nPairs; ni+=1) // 42
    {
        printf("///////////////////////////////////////////////////////////////////\n");
        printf("frameId:%d\n", ni);
        std::cout << "Left frame " << leftImagePaths[ni] << std::endl;
        std::cout << "Right frame " << rightImagePaths[ni] << std::endl;

        // Read image from file
        leftImage = cv::imread(leftImagePaths[ni].string(), 0);
        rightImage = cv::imread(rightImagePaths[ni].string(), 0);
        double tframe = vTimestamps[ni];

        if (leftImage.empty() || rightImage.empty()){
            cerr << endl << "Failed to load image at: "
                 << leftImagePaths[ni] << "/n" << rightImagePaths[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        if (mode == 0){
            sys.Run2(leftImage, rightImage, tframe, imageNames[ni].string());
        }
        else{
            sys.Run3(leftImage, rightImage, tframe, imageNames[ni].string());
        }

        // Check
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // vTimesTrack record the timing cost of this tracking.
        vTimesTrack[ni] = ttrack;
    }

    return 0;
}
