# Exoplanet Habitability

Datasets:

https://www.kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog
https://www.kaggle.com/datasets/dandandooo/images

Abstract:

With modern issues such as global warming, climate change and depletion of natural resources, combined with an increase in interest in exploring the expanses of space, the need to find a new home for the human race has exponentially increased over the years. Presently, the likelihood that we must abandon Earth for a new home has become far more likely, and in this current climate, finding a habitable exoplanet (planet outside our solar system) is imperative. Due to this circumstance, we have decided to develop an algorithm to take a user's input of data discovered about an exoplanet and determine whether it is habitable. This project can help save the human race as scientists and astrophysicists can quickly determine if a planet is potentially habitable to find a new home for humans quicker. 

Introduction:

This application is designed to predict whether an exoplanet is habitable or not based on input data. We use decision trees to find the columns which best represent the data the most.  We then feed them as inputs in a FCNN to classify whether they are habitable (suitable for life) or not.  Then we make an AI-generated image of the planet based on the data we had using a GAN.

Related Works:

We used concepts we learned in class such as FCNNs, CNNs and Decision Trees. As we didn’t know much python nor any AI/ML information before this course, most of the code is stuff we learnt in this course barring GANs and GUIs, which we needed to learn about in our own time. Therefore, everything we learnt in class was useful for this project as we needed a strong grasp on neural networks, image classification and visual interfaces.

Results:

The final product has the user entering all the required parameters that relate to characteristics of the exoplanet, and classifies it to either Habitable or Not Habitable. Taking the characteristics, it also generates an image of what the exoplanet might look like. All of this is combined in one visually appealing window using a GUI, Tkinter.
