#ifndef PCA_HPP
#define PCA_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS 
#define EIGEN_USE_LAPACK
#endif
#include <Eigen/Dense>

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using LossFunction=std::function<float(const MatrixXf&,const MatrixXf&)>;

/**
 * @brief PCA Analysis handler
*/
class PCAHandler{
private:
  std::string store_path;
  
  const MatrixXf& data; //< All the data (training only used)
  MatrixXf covariance_matrix;

  std::vector<std::pair<float,VectorXf>> eigen_pairs;
  float total_energy;

  MatrixXf principal_components; //< The result of the PCA
  LossFunction loss_func; //< The way to compare the input and output



public:
  PCAHandler(const MatrixXf& data):data(data){};
  void setStorePath(std::string store_path){this->store_path=store_path;}

  void createCovarianceMatrix();

  void createEigenPairs();

  void createPrincipalComponents(int components_number);

  void createPrincipalComponents(float info_percentage);

  float reconstructionMSE(const MatrixXf& dataset);

  float info_percentage(int components_number);

  void store();

  void load();
};

#endif 
