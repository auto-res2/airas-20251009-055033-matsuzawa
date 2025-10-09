
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
{}
---

# Dataset Card for Tiny-ImageNet-C

<!-- Provide a quick summary of the dataset. -->

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

In Tiny ImageNet-C, there are 75,109 corrupted images derived from the original Tiny ImageNet dataset. The images are affected by two different corruption types at five severity levels.

- **License:** CC BY 4.0

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Homepage:** https://github.com/hendrycks/robustness
- **Paper:** Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

Total images: 75,109

Classes: 200 categories

Splits:

- **Test:** 75,109 images

Image specs: JPEG format, 64×64 pixels, RGB

## Example Usage
Below is a quick example of how to load this dataset via the Hugging Face Datasets library.
```
from datasets import load_dataset  

# Load the dataset  
dataset = load_dataset("randall-lab/tiny-imagenet-c", split="test", trust_remote_code=True)

# Access a sample from the dataset  
example = dataset[0]  
image = example["image"]  
label = example["label"]  

image.show()  # Display the image  
print(f"Label: {label}")
```

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@article{hendrycks2019benchmarking,
  title={Benchmarking neural network robustness to common corruptions and perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  journal={arXiv preprint arXiv:1903.12261},
  year={2019}
}

Output:
{
    "extracted_code": "from datasets import load_dataset\n\n# Load the dataset\ndataset = load_dataset(\"randall-lab/tiny-imagenet-c\", split=\"test\", trust_remote_code=True)\n\n# Access a sample from the dataset\nexample = dataset[0]\nimage = example[\"image\"]\nlabel = example[\"label\"]\n\nimage.show()  # Display the image\nprint(f\"Label: {label}\")"
}
