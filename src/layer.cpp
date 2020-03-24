#include <torch/torch.h>
#include <layer.h>

namespace gcn {

  GraphConvolutionImpl::GraphConvolutionImpl(int In_features, int Out_features, bool Bias) :
         in_features(In_features), out_features(Out_features) {
    weights = register_parameter("weights", torch::rand({in_features, out_features}, torch::dtype(torch::kFloat32).requires_grad(true)));
    if(Bias) {
      bias = register_parameter("bias", torch::rand(out_features, torch::dtype(torch::kFloat32).requires_grad(true)));
    }
    else {
      bias = register_parameter("bias", torch::Tensor(), false);  // TODO 
    }
    reset_parameters();
  }

  void GraphConvolutionImpl::reset_parameters(){
     float stdv = 1 / sqrt(weights.sizes()[1]);
  //   std::cout << "weights: " << weights << std::endl;
     torch::nn::init::uniform_(weights, -stdv, stdv);
 //    std::cout << "stdv: " << stdv << std::endl;
     if(!bias.defined()) {
        bias =  register_parameter("bias", torch::Tensor(), false);
     }
    // std::cout << "bias: " << bias << std::endl;
  }

  torch::Tensor GraphConvolutionImpl::forward(torch::Tensor input, torch::Tensor adj) {
    // support = torch.mm(input, self.weight)
    // output = torch.spmm(adj, support)
    torch::Tensor support = torch::mm(input, weights);
    torch::Tensor output = at::hspmm(adj, support);   // torch::spmm is not available yet
  //  std::cout << "output: " << output << std::endl;
    if(bias.defined()){
    //  bias.unsqueeze(-1);
      torch::Tensor bias_expand = bias.expand_as(output);     // resize the bias
      return bias_expand + output;
    } else {
      return output;
    }
  }
}


  

/*
int main() {
//  std::cout << tensor << std::endl;
  GraphConvolution gcn(10, 4, true);
  torch::Tensor input = torch::randn({20, 10});
 // torch::Tensor adj;
 // the data type must be kLong!!!!
  torch::Tensor indices = at::tensor({0,0,1,2,2,3,0,3}, at::dtype(at::kLong)).view({2, 4});   
  torch::Tensor values = at::tensor({1,2,1,3}, at::dtype(at::kFloat)).view({4});

  torch::Tensor adj = at::sparse_coo_tensor(indices, values, {20,20});
  // std::cout << "adj: " << adj << std::endl;
  torch::Tensor output = gcn.forward(input, adj);
  std::cout << "output: " << output << std::endl;
   }
   */