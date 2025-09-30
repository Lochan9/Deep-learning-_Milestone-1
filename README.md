# CelebA Face Identification with CNNs

This project explores face identification using the **CelebA dataset**.  
We train and compare two deep learning models:  
- A **custom Simple CNN**  
- **ResNet18 (transfer learning)**  

The system can predict the **identity ID** of a given face and retrieve all images belonging to the same person.

---

## 📂 Project Structure

Deep-learning-_Milestone-1/
│
├── src/ # source code
│ ├── train.py # training both CNN and ResNet18
│ ├── evaluate.py # evaluation and metrics
│ ├── test.py # test on custom image + retrieve same identity
│
├── models/ # trained models (.pth files, optional if small)
│
├── data/ # dataset (not included in repo, see setup below)
│
├── requirements.txt # dependencies
├── README.md # project documentation
└── .gitignore
