# ctds-search-model
Work done with Center for Translational Data Science from September 2023 - Present.

This goal of this project is to design a free-text search tool for chest x-ray images within the the MIDRIC (Medical Imaging and Data Resource Center) database. 

## Background

MIDRC is an open-access data commons that contains hundreds of thousands of medical images and associated data, including chest x-ray data. Currently, users can search and filter through this data using curated metadata.

Our tool employs embedding-based search utilizing pre-trained contrastive vision-language models specifically for chest x-ray data. This allows users to locate specific data based on arbitrary criteria. For example, the search query "list of pneumonia images" should enable a user to retrieve a list of chest x-ray images of patients with a positive pneumonia diagnosis.

Our research builds on a variety of previous literature, notably including OpenAI's Contrastive Language-Image Pre-training (CLIP) method. CLIP popularized contrastive learning, enabling the alignment of different data modalities. By pre-training models with text and image pairs, it prioritizes the similarity between these pairs. This approach is applicable to our current work, as our tool compares a user's search query with existing images in the MIDRC database.

Another notable method that our research utilizes is the retrieval element from Retrieval Augmented Generation (RAG). RAG is a framework designed to augment language model generations using examples. The framework retrieves some top-k documents based on embedding similarity (e.g., cosine similarity) of a query term to previously vectorized documents/examples from a predefined corpus. We use a similar process to compare the search query embedding to a top-k number of vectorized images. In this case, our corpus is the image database, while one document would be a single image in the database.

For model development, we utilized Microsoft's BioViL and BioViL-T models, where contrastive learning approaches are applied to radiomic data over the MIMIC-CXR dataset. BioViL uses a CXR-BERT text encoder, pre-trained on a large corpus of radiology notes using Masked Language Modeling (MLM). For image encoding, BioViL uses a convolutional neural network (CNN). A self-supervised vision-language processing (VLP) approach for paired biomedical data was used to train the model. BioViL-T is an updated version of the BioViL model, incorporating prior images for temporal information. The image encoder is also updated to a CNN-Transformer hybrid multi-image encoder to accomplish this.

The MIMIC-CXR dataset that the BioViL model was trained on is a collection of chest x-ray-radiology report pairs collected from the Beth Israel Deaconess Medical Center in Boston, MA. In addition to the report-image pairs themselves, the CheXpert and CheXbert models were used to extract multi-category labels from the radiology notes. This provides report-image pairs for ground-truth comparison to evaluate the accuracy of our model's retrieval system.

Another notable work is Contrastive Visual Representation Learning from Text (ConVIRT), which utilizes contrastive pretraining and similarly evaluated text-image retrieval. The ConVIRT model was trained with the same MIMIC-CXR dataset and tested using a CheXpert 8x200 dataset. ConVIRT leverages the power of contrastive learning to create robust visual representations from paired text and image data, enhancing the accuracy of retrieval tasks. The method has shown significant improvements in text-to-image and image-to-text retrieval performance

## Specific Aims:
**Aim 1:** Evaluate accuracy of embedding based retrieval of imaging studies using BioViL-T for single labels with MIMIC-CXR.

**Aim 2:** Develop method to perform and evaluate multi-query expansion of search terms with MIMIC-CXR.

**Aim 3:** Adapt BioViL-T image and text models to MIDRC using hybrid data. 

