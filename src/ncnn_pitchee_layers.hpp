#ifndef PITCHEE_NCNN_PITCHEE_LAYERS_HPP
#define PITCHEE_NCNN_PITCHEE_LAYERS_HPP

namespace ncnn {
class Net;
}

namespace pitchee {

void register_ncnn_layers(ncnn::Net& net);

}  // namespace pitchee

#endif
