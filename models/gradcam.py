import time
import torch
import torch.nn.functional as F


def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 or v7 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


class YOLOV7GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        """Sets the fwd and bwrd hooks to get input/output gradients at a given layer  
        Args:
            model (torch): yolov7 model
            layer_name (string): target layer to be analyzed (layer ID and TYPE separated by underline)
            img_size (tuple, optional): image size. Defaults to (640, 640).
        Returns:
            None: 
        """        
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        # Identify the target layer in the model modules
        target_layer = find_yolo_layer(self.model, layer_name)
        
        # Obtain the input and output of each layer in the forward process to compare whether the hook is recorded correctly
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device)) # fwd pass with zero tensor with img dims
        print('[INFO] INIT: saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        
        # Perform a fwd pass on the input image
        tic = time.time()
        preds, logits = self.model(input_img)
        print("[INFO] FWD: model-forward took: ", round(time.time() - tic, 4), 'seconds')
        
        # Loop on each detection
        # for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
        logit, cls, cls_name = logits[0][0], preds[1][0][0], preds[2][0][0]
            
        # Select class of interest:  highest proba class or selected class id
        if class_idx:
            score = logit[class_idx]
        else:
            score = logit.max()
            
        # zero gradients
        self.model.zero_grad()
        
        # For the output of a certain category of the model, perform backpropagation
        tic = time.time()
        score.backward(retain_graph=True)
        print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
        
        # step1: Evaluate the gradients the selected output class with regard to the feature maps at the selected layer
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size() # torch.Size([1, 1024, 12, 20]) k:activation map, (u,v): size of activation map

        # step2: Compute the neuron importance weights alpha by doing a global average pool of the class activated gradients 
        # over the width and height dimension of the feature maps 
        alpha = gradients.view(b, k, -1).mean(2) # torch.Size([1, 1024, 240]) averaged over dim 2
        
        # Dimension adjustment, for subsequent multiplication of the output value of the target layer point by point
        weights = alpha.view(b, k, 1, 1) # torch.Size([1, 1024, 1, 1])
        
        # step3a: Perform a weighted combination of the feature map activation using the neuron importance weights 
        # (pixel wise sum of weighted activation values at the target layer)
        saliency_map = (weights * activations).sum(1, keepdim=True)

        # step 3b: Remove negative values - ReLu activation
        saliency_map = F.relu(saliency_map)

        # rescal saliency map
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        
        # Normalize saliency map
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        print(f"[INFO] Min/Max saliency: {saliency_map_min}/{saliency_map_max}")
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        saliency_maps.append(saliency_map)
            
        return saliency_maps, logits, preds

    def __call__(self, input_img, class_idx):
        return self.forward(input_img, class_idx)


class YOLOV7GradCAMPP(YOLOV7GradCAM):
    
    def __init__(self, model, layer_name, img_size=(640, 640)):
        super(YOLOV7GradCAMPP, self).__init__(model, layer_name, img_size)

    def forward(self, input_img, class_idx=True):
        """_summary_
        Args:
            input_img (_type_): _description_
            class_idx (bool, optional): _description_. Defaults to True.
        Returns:
            _type_: _description_
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        
        # Perform a fwd pass on the input image
        tic = time.time()
        preds, logits = self.model(input_img)
        print("[INFO] INIT: model-forward took: ", round(time.time() - tic, 4), 'seconds')
        
        # Loop on each detection
        # for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
        logit, cls, cls_name = logits[0][0], preds[1][0][0], preds[2][0][0]
         
        # Select class of interest:  highest proba class or selected class id
        if class_idx:
            score = logit[class_idx]
        else:
            score = logit.max()
            
        # zero gradients
        self.model.zero_grad()         
        
        # perform backpropagation 
        tic = time.time()
        score.backward(retain_graph=True)
        print(f"[INFO] FWD: {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
        
        # Evaluate the gradients the selected output class with regard to the feature maps at the selected layer
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = gradients.size()

        # Compute weights as combination of  positive partial derivatives of selected convolutional layer feature maps with respect to a specific class score
        # source: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks" Aditya Chattopadhyay 2018
        alpha_num = gradients.pow(2)
        
        alpha_denom = gradients.pow(2).mul(2) + \
                        activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)            
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom)) # check for zero value prior to division
        alpha = alpha_num.div(alpha_denom + 1e-7)

        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        # Perform a weighted combination of the feature map activation using the neuron importance weights 
        # (pixel wise sum of weighted activation values at the target layer)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        
        # Remove negative values
        saliency_map = F.relu(saliency_map)
        
        # Resize to input image size
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        
        # Normalize saliency map
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        print(f"[INFO] Min/Max saliency: {saliency_map_min}/{saliency_map_max}")
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        saliency_maps.append(saliency_map)            

        return saliency_maps, logits, preds

    def __call__(self, input_img, class_idx):
        return self.forward(input_img, class_idx)
    