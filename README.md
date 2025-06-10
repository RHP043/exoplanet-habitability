# Exoplanet Habitability

## Datasets

- [PHL Exoplanet Catalog](https://www.kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog)
- [Exoplanet Images Dataset](https://www.kaggle.com/datasets/dandandooo/images)

---

## Abstract

With modern issues such as global warming, climate change, and depletion of natural resources—combined with increased interest in space exploration—the need to find a new home for humanity is becoming increasingly urgent. The likelihood that we must eventually abandon Earth has never been higher.

This project aims to assist in the search for habitable exoplanets—planets located outside our solar system. We have developed an algorithm that, given data about an exoplanet, predicts its habitability. By accelerating the classification process, this tool can help scientists and astrophysicists identify promising candidates for human habitation more efficiently.

---

## Introduction

This application predicts whether an exoplanet is **habitable** or **not** based on input data.

### Methodology:
- We use **decision trees** to identify the most significant features in the dataset.
- These selected features are passed into a **Fully Connected Neural Network (FCNN)** to classify the planet as habitable or not.
- Additionally, we generate an **AI-based image** of the exoplanet using a **Generative Adversarial Network (GAN)**.
- All interactions are integrated into a single user-friendly **Tkinter GUI**.

---

## Related Works

This project builds upon several core AI/ML concepts covered in class, including:

- **Fully Connected Neural Networks (FCNNs)**
- **Convolutional Neural Networks (CNNs)**
- **Decision Trees**

Although we started the course with little to no background in Python or AI/ML, we applied everything we learned in class to complete this project. GANs and GUI development with Tkinter were explored independently to enhance our application further.

---

## Results

The final product allows users to:

1. Enter key parameters related to the characteristics of an exoplanet.
2. Receive a classification of either **Habitable** or **Not Habitable**.
3. View a generated visual representation of what the exoplanet might look like, based on its input characteristics.

All these features are combined into a single, visually appealing interface built using **Tkinter**.

---
