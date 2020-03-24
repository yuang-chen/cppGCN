#include <torch/torch.h>
#include "layer.h"

namespace gcn {

class GCNImpl : public torch::nn::Module {
public:
    GCNImpl(int n_feat, int n_hid, int n_class, float drop_out);
    torch::Tensor forward(torch::Tensor x,torch::Tensor adj);

private:
    int nfeat;
    int nhid;
    int nclass;
    float dropout;

    torch::nn::ReLU relu;
    GraphConvolution layer1 = nullptr;
    GraphConvolution layer2 = nullptr;
};

}