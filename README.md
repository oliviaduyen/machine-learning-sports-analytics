# NBA Player Position Classification Project

A machine learning project that classifies NBA players into their positions (Point Guard, Shooting Guard, Small Forward, Power Forward, Center) based on their statistical performance metrics.

## 📋 Project Overview

This project implements a classification model to predict NBA player positions using statistical data from the 2023 regular season. The model uses a Naive Bayes classifier to distinguish between the five traditional basketball positions based on performance metrics like points, assists, rebounds, and other game statistics.

## 🏀 Dataset

- **Primary Dataset**: `nba_stats.csv` - Contains NBA player statistics for the 2023 regular season
- **Test Dataset**: `dummy_test.csv` - Sample test dataset for model validation
- **Data Source**: Basketball Reference (https://www.basketball-reference.com/)

### Features Used
The model uses the following statistical features for classification:
- `MP` - Minutes Played per game
- `FGA` - Field Goal Attempts per game
- `3PA` - 3-Point Attempts per game
- `2PA` - 2-Point Attempts per game
- `FTA` - Free Throw Attempts per game
- `ORB` - Offensive Rebounds per game
- `DRB` - Defensive Rebounds per game
- `AST` - Assists per game
- `STL` - Steals per game
- `BLK` - Blocks per game
- `TOV` - Turnovers per game
- `PTS` - Points per game

### Target Classes
- `PG` - Point Guard
- `SG` - Shooting Guard
- `SF` - Small Forward
- `PF` - Power Forward
- `C` - Center

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `scikit-learn` - Machine learning algorithms and evaluation
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization
  - `jupyter` - Interactive notebook environment
  - `graphviz` - Decision tree visualization

## 🚀 Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd data-mining-project2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter graphviz
```

## 📊 Model Implementation

### Approach
The project implements a **Gaussian Naive Bayes classifier** with the following methodology:

1. **Data Preprocessing**: Load and prepare NBA statistics data
2. **Feature Selection**: Use 12 key performance metrics
3. **Train-Test Split**: 80% training, 20% validation (stratified sampling)
4. **Model Training**: Fit Naive Bayes classifier
5. **Evaluation**: Performance metrics and confusion matrices
6. **Cross-Validation**: 10-fold stratified cross-validation

### Model Performance
The model achieves competitive accuracy in classifying NBA player positions based on their statistical profiles.



## 🏃‍♂️ How to Run

### Option 1: Python Script
```bash
python P2.py
```

### Option 2: Interactive Jupyter Notebook (Recommended)
```bash
jupyter notebook NBA_Position_Classification.ipynb
```

### Option 3: View in GitHub
The notebook will render directly in GitHub for easy viewing.

## 📈 Results

The model provides:
- Training and validation accuracy scores
- Confusion matrices for performance analysis
- Cross-validation results with 10-fold validation
- Feature importance insights for basketball position classification

## 🎯 Key Insights

- Different positions show distinct statistical patterns
- Guards (PG, SG) typically have higher assist rates
- Centers (C) and Power Forwards (PF) dominate rebounding statistics
- The model successfully captures these positional differences

## 📚 Learning Objectives

This project demonstrates:
- Machine learning classification techniques
- Data preprocessing and feature selection
- Model evaluation and validation methods
- Sports analytics applications
- Python data science ecosystem usage

### 📊 Project Highlights
- ✅ **Real-world Dataset**: 875+ NBA players from 2023 season
- ✅ **Robust Methodology**: Stratified sampling and cross-validation
- ✅ **Comprehensive Analysis**: EDA, modeling, and evaluation
- ✅ **Practical Application**: Sports analytics and player evaluation

*This project demonstrates practical machine learning applications in sports analytics, showcasing skills in data science, statistical modeling, and Python programming.*