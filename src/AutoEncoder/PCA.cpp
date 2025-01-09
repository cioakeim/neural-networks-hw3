#include "AutoEncoder/PCA.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <iostream>


void PCAHandler::createCovarianceMatrix(){
  covariance_matrix=(data*data.transpose())/(data.rows()-1);
}


void PCAHandler::createEigenPairs(){
  // Solve and get eigen values and vectors
  std::cout<<"Solving"<<std::endl;
  E::SelfAdjointEigenSolver<MatrixXf> eigen_solver(covariance_matrix);
  MatrixXf eigen_vectors = eigen_solver.eigenvectors(); // Columns are eigenvectors
  VectorXf eigen_values = eigen_solver.eigenvalues();   // Eigenvalues
  std::cout<<"Solved"<<std::endl;
  // Store and sort
  eigen_pairs.clear();
  for(int i=0;i<eigen_values.size();i++){
    eigen_pairs.emplace_back(eigen_values(i),eigen_vectors.col(i));
  }
  sort(eigen_pairs.rbegin(), eigen_pairs.rend(), [](const auto& a, const auto& b) {
        return a.first < b.first;});
  total_energy=eigen_values.sum();
}


void PCAHandler::createPrincipalComponents(int components_number){
  std::cout<<"Init"<<std::endl;
  principal_components=MatrixXf(data.rows(),components_number); 
  std::cout<<"Init"<<std::endl;
  for(int i=0;i<components_number;i++){
    principal_components.col(i)=eigen_pairs[i].second; 
  }
}


void PCAHandler::createPrincipalComponents(float info_percentage){
  int last_idx=0;
  float energy_accum=0;
  float energy_target=total_energy*info_percentage;
  // Accumulate until target is reached
  int eigen_size=eigen_pairs.size();
  for(last_idx=0;last_idx<eigen_size;last_idx++){
    energy_accum+=eigen_pairs[last_idx].first;
    if(energy_accum>energy_target)
      break;
  }
  createPrincipalComponents(last_idx+1);
  std::cout<<"Components number: "<<principal_components.cols()<<std::endl;
}


float PCAHandler::reconstructionMSE(const MatrixXf& dataset){
  MatrixXf reconstruction = principal_components*
    principal_components.transpose()*dataset;
  return (reconstruction-dataset).array().pow(2).mean();
}


E::MatrixXf PCAHandler::reconstruct(const MatrixXf& input){
  MatrixXf reconstruction = principal_components*
    principal_components.transpose()*input;
  return reconstruction;
}


float PCAHandler::info_percentage(int components_number){
  float energy=0;
  for(int i=0;i<components_number;i++){
    energy+=eigen_pairs[i].first;
  }
  return energy/total_energy;
}

