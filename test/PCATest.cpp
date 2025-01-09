#include "CommonLib/cifarHandlers.hpp"
#include "AutoEncoder/PCA.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/basicFuncs.hpp"



int main(){
  std::string dataset_path="../data/cifar-10-batches-bin";
  std::string store_path="../data/PCA2/";
  int training_size=50000;
  int test_size=10000;
  float info_percentage=0.99;

  EventTimer et;
  et.start("Load dataset");
  Cifar10Handler c10=Cifar10Handler(dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);

  
  std::vector<NormalizationParams> params;

  normalizeImageDataset(training_set.vectors, test_set.vectors, 3,
                        params);
  for(auto param : params){
    std::cout<<"Mean: "<<param.mean<<" Sigma: "<<param.sigma<<std::endl;
  }
  et.stop();

  std::vector<E::MatrixXf> print_samples;
  for(int i=0;i<5;i++){
    print_samples.push_back(training_set.vectors.col(i));
  }
  c10.printMultipleSamples(print_samples,params);

  PCAHandler pca(training_set.vectors);

  ensure_a_path_exists(store_path);
  std::ofstream log(store_path+"/log.csv");
  log<<"Components #,Info percentage,TrainMSE,TestMSE"<<"\n";

  pca.createCovarianceMatrix();
  pca.createEigenPairs();
  std::cout<<"Done"<<std::endl;

  int components_num=512;
  int test_samples_sz=10;
  pca.createPrincipalComponents(components_num);
  E::MatrixXf original=training_set.vectors.middleCols(0,test_samples_sz);
  E::MatrixXf reconstructed=pca.reconstruct(original);
  for(int i=0;i<test_samples_sz;i++){
    std::vector<E::MatrixXf> temp;
    temp.push_back(original.col(i));
    temp.push_back(reconstructed.col(i));
    c10.printMultipleSamples(temp,params);
  }


  /*
  int total_component_sz=training_set.vectors.rows();
  for(int comp_num=124;comp_num<=total_component_sz;comp_num+=124){
    std::cout<<"NUM: "<<comp_num<<std::endl;
    int bounded_comp=std::min(comp_num,total_component_sz);
    pca.createPrincipalComponents(bounded_comp);
    float J_test,J_train,info_percentage;
    J_train=pca.reconstructionMSE(training_set.vectors);
    J_test=pca.reconstructionMSE(test_set.vectors);
    info_percentage=pca.info_percentage(bounded_comp);
    log<<bounded_comp<<","<<info_percentage<<","
      <<J_train<<","<<J_test<<"\n";
  }
  */
  
  

  et.displayIntervals();
  et.writeToFile(store_path+"/timing_info.txt");
}
