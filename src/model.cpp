#include <torch/torch.h>
#include <iostream>
#include "model.h"



namespace gcn {

    GCNImpl::GCNImpl(int n_feat, int n_hid, int n_class, float drop_out):
        nfeat(n_feat), nhid(n_hid), nclass(n_class), dropout(drop_out) {
            layer1 = GraphConvolution(nfeat, nhid);
            layer2 = GraphConvolution(nhid, nclass);
            register_module("layer1", layer1);
            register_module("layer2", layer2);
        }

    torch::Tensor GCNImpl::forward(torch::Tensor x,torch::Tensor adj){
        x = torch::relu(layer1->forward(x, adj));
        x = torch::dropout(x, dropout,is_training());
        x = layer2->forward(x, adj);
        x = torch::log_softmax(x, 1);
        return x;
    }

TORCH_MODULE(GCN);
};


int main() {
//  std::cout << tensor << std::endl;

  torch::Tensor input = torch::randn({20, 10});
 // torch::Tensor adj;
 // the data type must be kLong!!!!
  torch::Tensor indices = at::tensor({0,0,1,2,2,3,0,3}, at::dtype(at::kLong)).view({2, 4});   
  torch::Tensor values = at::tensor({1,2,1,3}, at::dtype(at::kFloat)).view({4});

  torch::Tensor adj = at::sparse_coo_tensor(indices, values, {20,20});
  // std::cout << "adj: " << adj << std::endl;

  gcn::GCN net(10,4,5,0.5);
  torch::Tensor output = net->forward(input, adj);
  std::cout << "output: " << output << std::endl;
}