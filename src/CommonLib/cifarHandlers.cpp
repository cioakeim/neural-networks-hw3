#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicStructs.hpp"
#include <iostream>
#include <fstream>
#include <vector>

// Constructor
Cifar10Handler::Cifar10Handler(std::string dataset_folder_path)
    : dataset_folder_path(dataset_folder_path){
  // Init binary fstreams
  std::ostringstream oss;
  for(int i=0;i<5;i++){
    oss << dataset_folder_path << "/data_batch_" << i+1 << ".bin";
    this->batch_file_streams[i].open(oss.str(),std::ios::binary);
    if(!batch_file_streams[i].is_open()){
      std::cerr<<"Error in opening: "<<oss.str()<<std::endl;
      exit(1);
    }
    oss.str("");
  }
  // Init test batch 
  oss << dataset_folder_path << "/test_batch.bin";
  this->test_batch_stream.open(oss.str(),std::ios::binary);
  if(!test_batch_stream.is_open()){
    std::cerr<<"Error in opening: "<<oss.str()<<std::endl;
    exit(1);
  }
  oss.str("");
  // Init lut 
  oss<< dataset_folder_path << "/batches.meta.txt";
  std::ifstream class_file(oss.str());
  if(!class_file.is_open()){
    std::cerr<<"Error in opening: "<<oss.str()<<std::endl;
    exit(1);
  }
  for(int i=0;i<10;i++){
    std::getline(class_file,this->id_to_class_name_lut[i]);
  }
  class_file.close();
};

// Destructor
Cifar10Handler::~Cifar10Handler(){
  for(int i=0;i<5;i++){
    batch_file_streams[i].close();
  }
  test_batch_stream.close();
}

// Get a new entry
int Cifar10Handler::getBatchEntry(int batch_id,SamplePoint& output){
  std::ifstream& current_stream=(batch_id==-1)?
                                  this->test_batch_stream:
                                  this->batch_file_streams[batch_id]; 

  // Get vector's dimensions right
  output.vector.resize(this->space_dimension); 
  // Get class_id 
  current_stream.read(reinterpret_cast<char*>(&output.label),1);
  if(!current_stream){
    return -1;
  }
  // Get vector in byte format.
  std::vector<uint8_t> buffer(this->space_dimension);
  current_stream.read(reinterpret_cast<char*>(buffer.data()),
                      this->space_dimension);
  // Normalize to float.
  for(int i=0;i<this->space_dimension;i++){
    output.vector[i]=static_cast<float>(buffer[i])/255.0f;
  }

  return 0;
}


std::vector<SamplePoint> Cifar10Handler::getTrainingList(int count){
  std::vector<SamplePoint> sample_list;
  SamplePoint temp;
  // Look for all batches
  int current_batch=0;
  for(int i=0;i<count;i++){
    // If this batch is empty go to the next.
    if(getBatchEntry(current_batch,temp)!=0){
      i--; // This wasn't a real entry
      current_batch++;
      // EOF, just return
      if(current_batch==5){
        break;
      }
      // There is another batch
      continue;
    }
    sample_list.push_back(temp);
  }
  return sample_list;
}


std::vector<SamplePoint> Cifar10Handler::getTestList(int count){
  std::vector<SamplePoint> sample_list;
  SamplePoint temp;
  for(int i=0;i<count;i++){
    // If EOF, end function
    if(getBatchEntry(-1,temp)!=0){
      break;
    }
    sample_list.push_back(temp);
  }
  return sample_list;
}


std::string Cifar10Handler::getClassName(int class_id){
  if(class_id>=0 && class_id<10)
    return this->id_to_class_name_lut[class_id];
  return "";
}

#ifdef WITH_OPENCV

#include <opencv4/opencv2/opencv.hpp>

void Cifar10Handler::printSample(const E::MatrixXf& sample,
                                 std::vector<NormalizationParams> params){
  static int image_cnt=0;
  E::MatrixXf denorm_sample=sample;
  denormalizeSamples(denorm_sample,params);
  const int height=32,width=32;
  // H X W uint8_t 3 channel matrix
  cv::Mat image(height,width,CV_8UC3);
  // Populate image 
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      // Index at flat array 
      int idx=i*height+j;
      // Get all 3 channels (BGR format)
      image.at<cv::Vec3b>(i,j)[0]=static_cast<uint8_t>(255*denorm_sample(idx+2*height*width,0)); // B
      image.at<cv::Vec3b>(i,j)[1]=static_cast<uint8_t>(255*denorm_sample(idx+height*width,0)); // G
      image.at<cv::Vec3b>(i,j)[2]=static_cast<uint8_t>(255*denorm_sample(idx,0)); // R
    }
  }
  // Resize image
  cv::Mat largerImage;
  cv::resize(image, largerImage, cv::Size(), 10.0, 10.0);

  cv::imshow("Sample"+std::to_string(image_cnt),largerImage);
  std::vector<int> save_params = {cv::IMWRITE_JPEG_QUALITY, 95}; // Quality 95
  cv::imwrite("output_"+std::to_string(image_cnt++)+".png", image, save_params);

}


void Cifar10Handler::printMultipleSamples(std::vector<E::MatrixXf> samples,
                                          std::vector<NormalizationParams> params){
  for(auto& sample: samples){
    printSample(sample,params);
  }
  std::cout<<"Here"<<std::endl;
  while (true) {
    char key = (char)cv::waitKey(1); // Check for key press with a delay of 1 ms
    if (key == 'q') {
      break;
    }
  }
  cv::destroyAllWindows();
}
#else
void Cifar10Handler::printSample(const E::MatrixXf& sample,
                                 std::vector<NormalizationParams> params){
    std::cout<<"No image support..."<<std::endl;
}
  
void Cifar10Handler::printMultipleSamples(std::vector<E::MatrixXf> samples,
                                          std::vector<NormalizationParams> params){
    std::cout<<"No image support..."<<std::endl;
}
#endif


SampleMatrix Cifar10Handler::getTrainingMatrix(int sample_count){
  std::vector<SamplePoint> data=this->getTrainingList(sample_count);

  SampleMatrix result;
  result.vectors=E::MatrixXf(data[0].vector.size(),data.size());
  result.labels=E::VectorXi(data.size());

  const int size=data.size();
  for(int i=0;i<size;i++){
    result.vectors.col(i)=data[i].vector;
    result.labels(i)=data[i].label;
  }
  return result;
}


SampleMatrix Cifar10Handler::getTestMatrix(int sample_count){
  std::vector<SamplePoint> data=this->getTestList(sample_count);

  SampleMatrix result;
  result.vectors=E::MatrixXf(data[0].vector.size(),data.size());
  result.labels=E::VectorXi(data.size());

  const int size=data.size();
  for(int i=0;i<size;i++){
    result.vectors.col(i)=data[i].vector;
    result.labels(i)=data[i].label;
  }
  return result;

}
















