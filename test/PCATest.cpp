#include "CommonLib/cifarHandlers.hpp"
#include "AutoEncoder/PCA.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/basicFuncs.hpp"



int main(){
  std::string dataset_path="../data/cifar-10-batches-bin";
  int training_size=50000;
  int test_size=1000;
  float info_percentage=0.9;

  EventTimer et;
  et.start("Load dataset");
  Cifar10Handler c10=Cifar10Handler(dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);
  /*
  normalizeDataset(training_set.vectors,
                   test_set.vectors);
  */
  normalizeImageDataset(training_set.vectors, test_set.vectors, 3);
  et.stop();


  PCAHandler pca(training_set.vectors);


  et.start("Convariance matrix");
  pca.createCovarianceMatrix();
  et.stop();
  std::cout<<"Done"<<std::endl;

  et.start("Eigen value solution");
  pca.createEigenPairs();
  et.stop();
  std::cout<<"Done"<<std::endl;

  et.start("Get principal components");
  pca.createPrincipalComponents(info_percentage);
  et.stop();
  std::cout<<"Done"<<std::endl;

  et.displayIntervals();
}
