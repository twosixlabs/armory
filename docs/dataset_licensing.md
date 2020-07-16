## Dataset Licensing

Armory datasets are either licensed or available in accordance to the fair use 
exception to copyright infringement. The passthrough license available to use 
or redistribution of the datasets for the licensed datasets is the Creative 
Commons 4.01 ShareAlike license.

## Original Licenses

| Dataset | Original license |
|:-:|:-:|
| MNIST | [Creative Commons Attribution-Share Alike 3.0](http://creativecommons.org/licenses/by-sa/3.0/) |  
| CIFAR-10 | [MIT](https://peltarion.com/knowledge-center/documentation/terms/dataset-licenses/cifar-10) |  
| Digit | [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) |  
| Librispeech | [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/) |
| GTSRB | [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Imagenette | [Apache 2.0](https://github.com/fastai/imagenette/blob/master/LICENSE) |  
| UCF101 | General copyright |
| RESISC45 | General copyright |

## Attributions

Note: attribution material can be removed upon request to the extent reasonably 
practicable. Please direct inquiries to <armory@twosixlabs.com>.

### MNIST
|Attribution                   |              |  
|------------------------------|--------------|
| Creator/attribution parties  | Yann LeCun and Corinna Cortes |
| Copyright notice             | Copyright &copy; 1998 by Yann LeCun and Corinna Cortes |
| Public license notice        | See (#original-licenses) |
| Disclaimer notice            | UNLESS OTHERWISE MUTUALLY AGREED TO BY THE PARTIES IN WRITING, LICENSOR OFFERS THE WORK AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE WORK, EXPRESS, IMPLIED, STATUTORY OR OTHERWISE, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTIBILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OF ABSENCE OF ERRORS, WHETHER OR NOT DISCOVERABLE. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION OF IMPLIED WARRANTIES, SO SUCH EXCLUSION MAY NOT APPLY TO YOU. |
| Dataset link | http://yann.lecun.com/exdb/mnist/ |
| License link | See [Original Licenses](#original-licenses) |
| Citation | LeCun, Yann, Corinna Cortes, and Christopher JC Burges. "The MNIST database of handwritten digits, 1998." URL http://yann.lecun.com/exdb/mnist 10, no. 34 (1998): 14. |
| Modification | (Slight) Representation of images as binary tensors |

### CIFAR-10
|Attribution                   |              |  
|------------------------------|--------------|
| Creator/attribution parties  | Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton |
| Copyright notice             | Not Available |
| Public license notice        | See (#original-licenses) |
| Disclaimer notice            | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. |
| Dataset link | https://www.cs.toronto.edu/~kriz/cifar.html |
| License link | See [Original Licenses](#original-licenses) |
| Citation | Krizhevsky, Alex. "Learning Multiple Layers of Features from Tiny Images." URL https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf, (2009). |
| Modification | (Slight) Representation of images as binary tensors |

## Fair use notes for RESISC-45 and UCF101
* Two Six Labs does not charge users for access to the Armory repository, 
nor the datasets therein, nor does it derive a profit directly from use of the 
data sets.
* Two Six Labs is not merely republishing the original datasets. The 
datasets have undergone transformative changes, specifically they have been 
repackaged to be integrated with Tensorflow Datasets. This repackaging 
includes, but is not limited to, processing images from compressed formats into 
binary tensors as well as decoding audio and video files. Further, Two Six Labs 
has published derived adversarial datasets that modify the original images with 
small perturbations that are crafted to fool machine learning models for both 
the RESISC-45 and UCF101 datasets.
* Two Six Labs uses these data sets within Armory, however there are 
other additional datasets present, as well as multiple other features present 
in Armory beyond providing datasets.
* Two Size Labs provides public benefit through the public distribution 
of the Armory framework to evaluate machine learning models. This material is 
based upon work supported by the Defense Advanced Research Projects Agency 
(DARPA) under Contract No. HR001120C0114. Note: Any opinions, findings and 
conclusions or recommendations expressed in this material are those of the 
author(s) and do not necessarily reflect the views of the Defense Advanced 
Research Projects Agency (DARPA).