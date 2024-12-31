#include "CommonLib/basicFuncs.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <omp.h>

namespace fs=std::filesystem;

void ensure_a_path_exists(std::string file_path){
  fs::path dir(file_path);
  if(!fs::exists(dir)){
    fs::create_directories(dir);
  }
}

std::string create_network_folder(std::string folder_path){
  ensure_a_path_exists(folder_path);
  int current_entry=0;
  while(fs::exists(folder_path+"/network_"+std::to_string(current_entry))){
    current_entry++;
  }
  std::string network_root=folder_path+"/network_"+std::to_string(current_entry);
  fs::create_directory(network_root);
  return network_root;
}

// Auxiliary
int count_directories_in_path(const fs::path& path) {
  int dir_count = 0;
  // Check if the given path exists and is a directory
  if (fs::exists(path) && fs::is_directory(path)) {
    // Iterate through the directory entries
    for (const auto& entry : fs::directory_iterator(path)) {
      // Increment the count for each directory (since we know all entries are directories)
      if(fs::is_directory(entry.path()))
        dir_count++;
    }
  }
  return dir_count;
}


E::MatrixXf loadMatrixFromFile(const std::string file_path){
  std::ifstream file(file_path);
  if(!file.is_open()){
    std::cerr<<"Error in loading: "<<file_path<<std::endl;
    exit(1);
  }
  int rows,cols;
  file>>rows>>cols;
  E::MatrixXf matrix(rows,cols);
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      file>>matrix(i,j);
    }
  }
  return matrix;
}


void storeMatrixToFile(const std::string file_path,
                       const E::MatrixXf matrix){
  std::ofstream file(file_path);
  if(file.is_open()){
    file<<matrix.rows()<<" "<<matrix.cols()<<"\n"; // Dimensions
    file<<matrix<<"\n";
  }
  else{
    std::cerr<<"Error in storing: "<<file_path<<std::endl;
    exit(1);
  }
}


void shuffleDatasetInPlace(SampleMatrix& set){
  const int training_size=set.vectors.cols();
  // Shuffle training set 
   // Generate a random permutation of column indices
  std::vector<int> indices(training_size);
  std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, ..., cols-1
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  for(int i=0;i<training_size;i++){
    set.vectors.col(i).swap(set.vectors.col(indices[i]));
    int temp=set.labels[i];
    set.labels[i]=set.labels(indices[i]);
    set.labels[indices[i]]=temp;
  }
}


void normalizeDataset(E::MatrixXf& training_set,
                      E::MatrixXf& test_set){
  float train_sz=training_set.cols();
  float test_sz=test_set.cols();
  float mean=(train_sz*training_set.mean()+test_sz*test_set.mean())
              /(train_sz+test_sz);
  training_set.array()-=mean;
  test_set.array()-=mean;
  float train_sigma=sqrt(training_set.array().pow(2).mean());
  float test_sigma=sqrt(test_set.array().pow(2).mean());
  float sigma=(train_sz*train_sigma+test_sz*test_sigma)/
              (train_sz+test_sz);
  training_set.array()/=sigma;
  test_set.array()/=sigma;
}


void normalizeImageDataset(E::MatrixXf& training_set,
                           E::MatrixXf& test_set,
                           int channel_number){
  int pixels=training_set.rows()/channel_number;
  int training_sz=training_set.cols();
  int test_sz=test_set.cols();
  for(int i=0;i<training_set.rows();i+=pixels){
    E::MatrixXf train_channel=training_set.block(i,0,pixels,training_sz);
    E::MatrixXf test_channel=test_set.block(i,0,pixels,test_sz);
    normalizeDataset(train_channel,test_channel);
    training_set.block(i,0,pixels,training_sz)=train_channel;
    test_set.block(i,0,pixels,test_sz)=test_channel;
  }
}


std::vector<int> stringToVector(std::string str){
    std::vector<int> result;

    // Use a stringstream to split the input by commas
    std::stringstream ss(str);
    std::string token;
    // Extract integers from the comma-separated string
    while (std::getline(ss, token, ',')) {
        try {
            // Convert each token to an integer and store it in the vector
            result.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr<<"Invalid argument: "<< token <<" is not a valid integer."<< std::endl;
            exit(1);
        }
    }
    return result;
}


void NaNcheck(const E::MatrixXf& mat,std::string error_label){
  if(mat.array().isNaN().cast<int>().maxCoeff()>0){
    std::cout<<"NAN FOUND: "<<error_label<<std::endl;
  }
  else{
    std::cout<<error_label<<": "<<mat.array().pow(2).mean()<<std::endl;
  }
}


