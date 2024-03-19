# Tip of the Tounge

***Rutvik Sayankar, Charlie Plonski, Aahan Thapliyal***

Our team has observed a recurring issue amongst our members where we tend to struggle with recalling the names of individuals we encounter. As a remedy, we endeavor to develop an Android application that addresses this predicament. Our project comprises three facets. We aim to design an app that can track a user’s face in real time. Secondly, we aim to recognize and classify faces whilst tracking the user. Finally, we aimed to incorporate speech recognition capabilities, such that when the subject within the frame verbalizes their name, the application shall record and associate the name with their corresponding facial features. We intend to construct the Fisherfaces & Eigenfaces algorithm from scratch, whilst utilizing native Android libraries for the audio processing component of our project.

We refered to the following papers when creating this project:
1. P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman, “Eigenfaces vs. fisherfaces: Recognition using class specific linear projection,” IEEE Transactions on pattern analysis and machine intelligence, vol. 19, no. 7, pp. 711–720, 1997.
2. M. Turk and A. Pentland, “Eigenfaces for recognition,” Journal of cognitive neuroscience, vol. 3, no. 1, pp. 71–86, 1991.

Hardware Used:
– Lenovo X606V Smart Tab M10 FHD Plus (4GB RAM, 128GB Storage)
– NVIDIA SHIELD Tablet K1 (2 GB RAM, 128 GB Storage)

Softwares Used:
– Android Studio (Electric.Eel| 2022.1.1) 
– Jupyter Notebook


## Initializing

1. Clone the repository
```
git clone https://github.com/AahanT/Tip_of_the_Tongue.git
```
2. We used Yale Face Database to generate the eignenface and fisherface matrix. This database could be found on http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html
3. Change `path/to/face/data` to the path of the face dataset and run ***GettingFaces.ipynb***
4. Load **acr_eigen.csv** and **meanandface.csv** into the Tablet
5. Create a new project in *Android Studio* and replace the **activity_main.xml** with the **activity_main.xml** stored in the android folder
6. Build and Flash the *Android Studio* project

