#include <torch/torch.h>
#include <iostream>

namespace gcn {
class GraphConvolutionImpl : public  torch::nn::Module {
public:
    GraphConvolutionImpl(int In_features, int Out_features, bool Bias = true);
    torch::Tensor forward(torch::Tensor input, torch::Tensor adj);

private:
    int in_features;
    int out_features;
    torch::Tensor weights;
    torch::Tensor bias;
    void reset_parameters();
};

TORCH_MODULE(GraphConvolution);
}