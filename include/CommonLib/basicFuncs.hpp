#ifndef BASIC_FUNCS_HPP
#define BASIC_FUNCS_HPP 

#include <string>
#include <filesystem>
#include "CommonLib/basicStructs.hpp"

namespace fs=std::filesystem;
namespace E=Eigen;

// Ensures the path given exists
void ensure_a_path_exists(std::string file_path);

std::string create_network_folder(std::string folder_path);

int count_directories_in_path(const fs::path& path);

E::MatrixXf loadMatrixFromFile(const std::string file_path);

void storeMatrixToFile(const std::string file_path,
                       const E::MatrixXf matrix);

E::VectorXf loadVectorFromFile(const std::string file_path);

void storeVectorToFile(const std::string file_path,
                       const E::VectorXf vector);

void normalizeDataset(E::MatrixXf& training_set,
                      E::MatrixXf& test_set);

void normalizeImageDataset(E::MatrixXf& training_set,
                           E::MatrixXf& test_set,
                           int channel_number);

void shuffleDatasetInPlace(SampleMatrix& set);

std::vector<int> stringToVector(std::string str);

void NaNcheck(const E::MatrixXf& mat,std::string error_label);

int count_directories_by_prefix(std::string file_path, std::string keyword);




#endif
