# UAVid Semantic Segmentation

A project developed as part of the Neural Networks course at the National University of Kyiv-Mohyla Academy.

Notebooks are authored by Bohdan Prokhorov.

## Task

The goal was to pick any available dataset for an object detection or semantic segmentation project. I chose semantic segmentation since I already had some experience with object detection and wanted to explore something new.

The work was split into three parts. The first one involved building an EDA pipeline to collect basic statistics and develop an understanding of the data before diving in. The second part was about implementing two well-known neural network architectures from scratch and adapting them for this task. The third part focused on importing two current SOTA models and applying two methods of neural network explanation.

## Results

As a result, I put together an EDA notebook, implemented a Fully Convolutional Network (FCN) based on AlexNet and U-Net as the most influential architectures for this task. For SOTA models, I selected DeepLabV3 and SegFormer. Explainability was built for U-Net and SegFormer using GradCAM and Vanilla Gradients as the chosen explanation methods.

Train notebooks contain a full training pipeline with the `ray-tune` framework for hyperparameter optimization. Training loops are created from scratch and are similar from notebook to notebook. U-Net and AlexNet FCN are inside the `/models` folder, and a custom pixel-wise weighted focal loss is inside `/losses`. Explanation pipelines contain each method for each selected network and include visualizations of different layers and how the network "sees" different objects.

Every notebook is provided with conclusions (metrics, thoughts, etc.) about the architecture or approach.

## Final Conclusions

That was an interesting project where I've learned the details of implementing training pipelines, architectures, metrics for segmentation, model tuning using `ray-tune` for a `torch` project, and methods of explaining these networks. 